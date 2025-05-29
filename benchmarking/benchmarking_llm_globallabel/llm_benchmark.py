import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score

if __name__ == "__main__":
    openbio = pd.read_csv("CSV file for benchmark data from any LLM")['Overall'].tolist()
    test_df = pd.read_csv("<Path to test data split>")['Target'].tolist()

    fpr, tpr, roc_thresholds = roc_curve(test_df, openbio)
    auc_value                = roc_auc_score(test_df, openbio)
    upper_fpr_idx   = min(i for i, val in enumerate(fpr) if val >= 0.3)
    tar_at_far_103  = tpr[upper_fpr_idx]
    upper_fpr_idx   = min(i for i, val in enumerate(fpr) if val >= 0.2)
    tar_at_far_102  = tpr[upper_fpr_idx]

    np.savez("<npy saving path to store TPR and FPR>",
             fpr=fpr,
             tpr=tpr,
             thresholds=roc_thresholds,
             auc=auc_value,
             tar30=tar_at_far_103,
             tar20=tar_at_far_102)

    print(auc_value)
    print(tar_at_far_102)
    print(tar_at_far_103)