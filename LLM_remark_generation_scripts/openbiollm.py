import os
import transformers
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import pandas as pd
import json
import time
from huggingface_hub import login
from huggingface_hub import InferenceClient
login(token = "HF token")

class OpenBioLLM:
    def __init__(self, model_id, cache_dir, model_checkpoint, batch_size=4):
        self.model_id = model_id
        self.cache_dir = cache_dir
        self.model_checkpoint = model_checkpoint
        self.batch_size = batch_size
        self.tokenizer = self.load_tokenizer()
        self.model = self.load_model()
        self.pipeline = self.create_pipeline()
    def load_tokenizer(self):
        return transformers.AutoTokenizer.from_pretrained(self.model_id, padding_side='left', cache_dir=self.cache_dir)
    def load_model(self):
        return transformers.AutoModelForCausalLM.from_pretrained(
            self.model_checkpoint,
            dtype="auto",
            cache_dir=self.cache_dir,
            device_map="auto"
        )
    def create_pipeline(self):
        return transformers.pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            model_kwargs={"dtype": torch.bfloat16},
            batch_size=self.batch_size,
        )
    def generate_batch_responses(self, messages_batch, max_tokens=500):
        prompts = [
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            for messages in messages_batch
        ]
        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        outputs = self.pipeline(
            prompts,
            max_new_tokens=max_tokens,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.9,
            top_p=0.9,
        )
        return [output[0]["generated_text"][len(prompt):] for output, prompt in zip(outputs, prompts)]

class ClinicalDataProcessor:
    def __init__(self, knowledge_bank_path, clinical_summaries_path):
        self.knowledge_bank_path     = knowledge_bank_path
        self.clinical_summaries_path = clinical_summaries_path
        self.knowledge_bank          = self.load_knowledge_bank()
        self.clinical_summaries      = self.load_clinical_summaries()
        self.clinical_features       = [
            "Age", "BMI", "FEV1 Predicted", "DLCO Predicted", "Pack-Years Of Cigarette Use",
            "Gender", "Prior Cardiothoracic Surgery", "Preoperative Chemo - Current Malignancy",
            "Preoperative Thoracic Radiation Therapy", "Cigarette Smoking Indicator", "ECOG Score",
            "ASA Classification", "Tumor size", "TNM staging", "Procedure to be performed"]
    def load_knowledge_bank(self):
        with open(self.knowledge_bank_path, "r") as file:
            return file.read()
    def load_clinical_summaries(self):
        with open(self.clinical_summaries_path, "r") as file:
            return json.load(file)
    def create_prompt(self, summary):
        precontext  = "Suppose I have a summary of preoperative condition of a patient who will be going to have a lung cancer surgery, and need you to tell me some remarks, such that it is written by actual human doctor, stating why and why not the patient might have postoperative complications, keeping the relationship between preoperative features and the summary of the patient's preoperative condition into mind. The preoperative features taken into account is given as:\n" + str(self.clinical_features) + "\n" + """The postoperative complications cover the occurence of following conditions only:
        1. Pneumonia
        2. Adult Respiratory Distress Syndrome
        3. Atelectasis Requiring Bronchoscopy
        4. Bronchopleural Fistula
        5. Pneumothorax
        6. Air Leak Greater Than Five Days
        7. Unexpected Admission To ICU
        8. Empyema Requiring Treatment
        9. Initial Vent Support >48 Hours
        The contextual information on how different clinical features interact with each other with detailed relation and stratification, along with their contribution in postoperative complication is given below:\n""" + self.knowledge_bank + "\n" + "The summary of preoperative conditions of the patient is given as follows:\n"
        postcontext = """You are a thoracic-surgery clinician. Provide me a precise and to the point remark on why and why not this patient can have the above said postoperative complications. Stick to the following rules strictly for the output and the output format:\n
        1) Use ONLY the exact template below—no preamble, no extra text.\n
        2) Populate blanks strictly from the patient summary and the Knowledge Bank (KB).\n
        3) Clinical tone; no invented facts; tie ASSESSMENT reasons to KB or provided data.\n
        4) Write the assessment such that it keeps all high risk assessment sentiments (if any) together, moderate risk sentiments (if any) together and low risk sentiments (if any) together. For example: The patient will be at high risk of pneumonia, adult respiratory distress syndrome, atelectasis requiring bronchoscopy, and prolonged air leaks after lobectomy procedure given low BMI, current smoking status, and history of preoperative thoracic radiation therapy. However her high FEV1 and DLCO along with low pack years mitigates her risk of complication. Here it states the high risk predictors together and low risk predictors together.\n
        5) ASSESSMENT reason should be elaborate and somewhat detailed but not be too verbose, too long, too vague or having extra words. Provide the assessment considering all the clinical factors and KB data on how they interact with each other.\n
        OUTPUT FORMAT (use EXACTLY this)\n
        --------------------------------\n
        ___ years old, <Gender>, BMI of ___, <former/current/never> smoker with FEV1 of ___ and DLCO of ___,___ pack-year history, (had/did not had) prior thoracic surgery on same side, (had/did not had) preoperative chemotherapy specific to this cancer, (had/did not had) preoperative radiation specific to this cancer. ECOG __, ASA __, T__N__M__, Procedure ______. ASSESSMENT:\n The patient will be at <high/low/moderate> risk of x,y,z and so on complications after (procedure listed in the clinical summary) procedure given (clinical features contributing the assessment). (Clinical features) may (mitigate/increase) risk for some complications like x,y,z and so on.\n\n
        POPULATION RULES\n
        --------------------------------\n
        - Fill in the blanks with values from patient's clinical summary as provided.\n
        - Gender can be Male or Female, as given in clinical summary.\n
        - Smoking status only one of: former/current/never.\n
        - FEV1 and DLCO as percentages.\n
        - ECOG and ASA status from clinical summary.\n
        - TNM: use T?N?M? with available values.\n
        - ASSESSMENT is exactly one of Low/Moderate/High. After 'based on', list short phrases based on KB/data about what contributed towards postoperative complication and what did not. Don't be too verbose or vague, being precise and detailed. Be short but don't leave important detail. ASSESSMENT reason should be elaborate and somewhat detailed but not be too verbose, too long, too vague or having extra words.\n"""
        return precontext + summary + postcontext

def batch_iterate(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

def main():
    llm = OpenBioLLM(model_id   = "aaditya/OpenBioLLM-Llama3-70B",
                    cache_dir  = "",
                    model_checkpoint = "",
                    batch_size = 4)
    data_processor = ClinicalDataProcessor(
        knowledge_bank_path="../dataset/knowledge_bank.txt",
        clinical_summaries_path="Path to {train, val, test} summaries"
    )
    summaries  = list(data_processor.clinical_summaries.items())
    batch_size = llm.batch_size
    for batch in tqdm(batch_iterate(summaries, batch_size), total=len(summaries) // batch_size):
        batch_messages   = list()
        batch_record_ids = list()
        for record_id, record in batch:
            summary = record['Clinical_summary']
            text_prompt = data_processor.create_prompt(summary)
            messages = [
                {
                    "role": "system",
                    "content": "You are a thoracic-surgery clinician writing concise, professional remarks. Follow the rules and format below exactly."
                },
                {
                    "role": "user",
                    "content": text_prompt
                },
            ]
            batch_messages.append(messages)
            batch_record_ids.append(record_id)
        batch_responses = llm.generate_batch_responses(batch_messages)

        for record_id, response in zip(batch_record_ids, batch_responses):
            try:
                data_processor.clinical_summaries[record_id]['remarks'] = response[response.index("ASSESSMENT")+len("ASSESSMENT")+1:].replace("*","").strip()
                with open("Path to save remarks", "w") as js:
                    json.dump(data_processor.clinical_summaries, js, indent = 4)
            except:
                data_processor.clinical_summaries[record_id]['remarks'] = response.replace("*","").strip()
                with open("Path to save remarks", "w") as js:
                    json.dump(data_processor.clinical_summaries, js, indent = 4)

if __name__ == "__main__":
    main()