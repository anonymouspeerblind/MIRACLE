import transformers
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm
import pandas as pd
import json
import time
from huggingface_hub import login
from huggingface_hub import InferenceClient
login(token = "HF token")

if __name__ == "__main__":
    clinical_feats = ["Age", "BMI", "FEV1 Predicted", "DLCO Predicted","Pack-Years Of Cigarette Use", "Gender", "Prior Cardiothoracic Surgery", "Preoperative Chemo - Current Malignancy", "Preoperative Thoracic Radiation Therapy", "Cigarette Smoking Indicator", "ECOG Score", "ASA Classification", "Tumor size", "TNM staging", "Procedure to be performed"]
    with open("../dataset/knowledge_bank.txt", "r") as tx:
        KB = tx.read()
    with open("Path to {train, val, test} summaries", "r") as js:
        train_sum = json.load(js)
    
    precontext = "Suppose I have a summary of preoperative condition of a patient who will be going to have a lung cancer surgery, and need you to tell me some remarks, such that it is written by actual human doctor, stating why and why not the patient might have postoperative complications, keeping the relationship between preoperative features and the summary of the patient's preoperative condition into mind. The preoperative features taken into account is given as:\n" + str(clinical_feats) + "\n" + """The postoperative complications cover the occurence of following conditions only:
    1. Pneumonia
    2. Adult Respiratory Distress Syndrome
    3. Atelectasis Requiring Bronchoscopy
    4. Bronchopleural Fistula
    5. Pneumothorax
    6. Air Leak Greater Than Five Days
    7. Unexpected Admission To ICU
    8. Empyema Requiring Treatment
    9. Initial Vent Support >48 Hours
    The contextual information on how different clinical features interact with each other with detailed relation and stratification, along with their contribution in postoperative complication is given below:\n""" + KB + "\n" + "The summary of preoperative conditions of the patient is given as follows:\n"
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

    model_id  = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir = "")
    model     = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir = "",
        dtype="auto",
        device_map="auto",
        attn_implementation="flash_attention_2"
    )

    train_sum_copy = train_sum
    for record_id in tqdm(train_sum):
        summary     = train_sum[record_id]['Clinical_summary']
        text_prompt = precontext + summary + postcontext
        messages = [
        {"role": "system", "content": "You are a thoracic-surgery clinician writing concise, professional remarks. Follow the rules and format below exactly."},
        {"role": "user",   "content": text_prompt},
        ]
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt").to('cuda')

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        terminators = [tokenizer.eos_token_id]
        im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if im_end is not None and im_end != tokenizer.eos_token_id:
            terminators.append(im_end)
        
        outputs = model.generate(
            input_ids,
            max_new_tokens=500,
            do_sample=False,
            eos_token_id=terminators,
            pad_token_id=tokenizer.eos_token_id
        )
        response = outputs[0][input_ids.shape[-1]:]
        exact_remark = tokenizer.decode(response, skip_special_tokens=True)[tokenizer.decode(response, skip_special_tokens=True).index("</think>")+8:].strip()
        try:
            train_sum[record_id]['remarks'] = exact_remark[exact_remark.index("ASSESSMENT")+len("ASSESSMENT")+1:].replace("*","").strip()
            with open("Path to save remarks", "w") as js:
                json.dump(train_sum, js, indent = 4)
        except:
            train_sum[record_id]['remarks'] = exact_remark.replace("*","").strip()
            with open("Path to save remarks", "w") as js:
                json.dump(train_sum, js, indent = 4)