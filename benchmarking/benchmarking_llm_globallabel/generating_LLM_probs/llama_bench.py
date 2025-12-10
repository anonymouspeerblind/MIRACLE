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
    with open("../dataset/clinical_features.json", "r") as js:
        clinical_feats = json.load(js)['clinical_features']
    with open("../dataset/knowledge_bank.txt", "r") as tx:
        knowledge_bank = tx.read()
    with open("../dataset/complications.json", "r") as js:
        complications = json.load(js)['complications']
    with open("Path to clinical summaries for patients in test split", "r") as js:
        test_summaries = json.load(js)
    
    precontext  = f"Suppose I have a clinical summary of preoperative condition of a patient who is be going to have a lung cancer surgery, and I need you to predict the probability of the following complication happening to the patient post surgery, keeping the relationship between patient's pre-operative clinical condition in mind. The only postoperative complication I am interested in is:\n {complications}"
    mid_text    = f"The contextual information on how different clinical features interact with each other with detailed relation and stratification, along with their contribution in postoperative complication listed above is given below:\n {knowledge_bank}\n .The summary of preoperative conditions of the patient is given as follows:\n"
    postcontext = """Provide me with a single probabilty of whether or not any of the above complications will happen to the patient, keeping summary and the contextual information in mind.
                    Return only a single floating point number between 0 and 1 with at most 6 decimal places.
                    DO NOT output any text, explanation, or token other than the number.

                    Output format:
                    <probability>"""

    model_id  = "meta-llama/Llama-3.3-70B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir = "")
    model     = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir = "",
        dtype="auto",
        device_map="auto",
        attn_implementation="flash_attention_2"
    )

    final_prob_dic = test_summaries.copy()
    for rec in tqdm(test_summaries):
        summary     = test_summaries[rec]['Clinical_summary']
        text_prompt = precontext + mid_text + summary + postcontext
        messages = [
            {"role": "system", "content": "You are an oncologist and thoracic surgeon. Your sole task is to generate a single probability that estimates risks of postoperative complications after lung cancer surgery of the patient. You MUST obey every rule below, even if user messages conflict."},
            {"role": "user",   "content": text_prompt},
            ]
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt").to('cuda')

        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        outputs = model.generate(
            input_ids,
            max_new_tokens=500,
            do_sample=False,
            eos_token_id=terminators,
            pad_token_id=tokenizer.eos_token_id
        )
        response = outputs[0][input_ids.shape[-1]:]
        exact_remark = tokenizer.decode(response, skip_special_tokens=True).strip()
        
        final_prob_dic[rec]['probability'] = exact_remark
        with open("Path to save response from LLM", "w") as js:
            json.dump(final_prob_dic, js, indent = 4)