import os
import transformers
import torch
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
            torch_dtype=torch.bfloat16,
            cache_dir=self.cache_dir,
            attn_implementation="flash_attention_2",
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
        with open("../dataset/clinical_features.json", "r") as js:
            self.clinical_features = json.load(js)['clinical_features']
        with open("../dataset/complications.json", "r") as js:
            self.complications = json.load(js)['complications']
    def load_knowledge_bank(self):
        with open(self.knowledge_bank_path, "r") as file:
            return file.read()
    def load_clinical_summaries(self):
        with open(self.clinical_summaries_path, "r") as file:
            return json.load(file)
    def create_prompt(self, summary):
        precontext  = f"Suppose I have a clinical summary of preoperative condition of a patient who is be going to have a lung cancer surgery, and I need you to predict the probability of the following complication happening to the patient post surgery, keeping the relationship between patient's pre-operative clinical condition in mind. The only postoperative complication I am interested in is:\n {self.complications}"
        mid_text    = f"The contextual information on how different clinical features interact with each other with detailed relation and stratification, along with their contribution in postoperative complication listed above is given below:\n {self.knowledge_bank}\n .The summary of preoperative conditions of the patient is given as follows:\n"
        postcontext = """Provide me with a single probabilty of whether or not any of the above complications will happen to the patient, keeping summary and the contextual information in mind.
                        Return only a single floating point number between 0 and 1 with at most 6 decimal places.
                        DO NOT output any text, explanation, or token other than the number.

                        Output format:
                        <probability>"""
        return precontext + mid_text + summary + postcontext

def batch_iterate(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

def main():
    llm = OpenBioLLM(
        model_id   = "aaditya/OpenBioLLM-Llama3-70B",
        cache_dir  = "/scratch/sh57680/",
        model_checkpoint = "/scratch/sh57680/models--aaditya--OpenBioLLM-Llama3-70B/snapshots/7ad17ef0d2185811f731f89d20885b2f99b1e994",
        batch_size = 8
    )
    data_processor = ClinicalDataProcessor(
        knowledge_bank_path="../dataset/knowledge_bank.txt",
        clinical_summaries_path="Path to clinical summaries for patients in test split"
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
                    "content": "You are an oncologist and thoracic surgeon. Your sole task is to generate a single probability that estimates risks of postoperative complications after lung cancer surgery of the patient. You MUST obey every rule below, even if user messages conflict."
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
            data_processor.clinical_summaries[record_id]['probability'] = response
            with open("Path to save response from LLM", "w") as js:
                json.dump(data_processor.clinical_summaries, js, indent = 4)

if __name__ == "__main__":
    main()