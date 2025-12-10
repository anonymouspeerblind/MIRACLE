import os, json
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score

if __name__ == "__main__":
    rem = "type of LLM"
    with open(f"Path to probability json from LLM", "r") as js:
        consolidated_probs = json.load(js)
    test_df  = pd.read_csv("Path to test csv")
    test_dic = test_df.set_index("Record")["Target"].to_dict()

    pred_probs, target_probs = list(), list()
    for rec in consolidated_probs:
        pred_probs.append(float(consolidated_probs[rec]['probability']))
        target_probs.append(test_dic[float(rec)])

    fpr, tpr, roc_thresholds = roc_curve(target_probs, pred_probs)
    auc_value                = roc_auc_score(target_probs, pred_probs)
    upper_fpr_idx            = min(i for i, val in enumerate(fpr) if val >= 0.3)
    tar_at_far_103           = tpr[upper_fpr_idx]
    upper_fpr_idx            = min(i for i, val in enumerate(fpr) if val >= 0.2)
    tar_at_far_102           = tpr[upper_fpr_idx]

    np.savez(f"../benchmarking/benchmarking_llm_globallabel/{rem}.npz",
             fpr=fpr,
             tpr=tpr,
             thresholds=roc_thresholds,
             auc=auc_value,
             tar30=tar_at_far_103,
             tar20=tar_at_far_102)

    print(auc_value)
    print(tar_at_far_102)
    print(tar_at_far_103)