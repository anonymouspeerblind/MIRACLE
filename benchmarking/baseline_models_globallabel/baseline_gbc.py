import json
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve

with open("<path for list of clinical features>", "r") as js:
    clinical_feat = json.load(js)['clinical_features']
with open("<path for list of radiological features>", "r") as js:
    radiomic_feat = json.load(js)['radiomics_features']
clinical_feat.extend(radiomic_feat)
train_df = pd.read_csv("<Path to train dataset>")
test_df  = pd.read_csv("<Path to test dataset>")

X_train = train_df[clinical_feat]
y_train = train_df["Target"]
X_test  = test_df[clinical_feat]
y_test  = test_df["Target"]

gbc = GradientBoostingClassifier(
    n_estimators=100, 
    learning_rate=0.1, 
    max_depth=3,    
    random_state=42         # for reproducibility
)
gbc.fit(X_train, y_train)

y_pred_proba                = gbc.predict_proba(X_test)[:, 1]
fpr, tpr, roc_thresholds    = roc_curve(y_test, y_pred_proba)
auc_value                   = roc_auc_score(y_test, y_pred_proba)
upper_fpr_idx               = min(i for i, val in enumerate(fpr) if val >= 0.3)
tar_at_far_103              = tpr[upper_fpr_idx]
upper_fpr_idx               = min(i for i, val in enumerate(fpr) if val >= 0.2)
tar_at_far_102              = tpr[upper_fpr_idx]

np.savez("<saving path to npy matrix to store TPR and FPR>",
             fpr=fpr,
             tpr=tpr,
             thresholds=roc_thresholds,
             auc=auc_value,
             tar30=tar_at_far_103,
             tar20=tar_at_far_102)

print(f"AUC value: {auc_value}")
print(f"TAR@FAR=0.3: {tar_at_far_103}")
print(f"TAR@FAR=0.2: {tar_at_far_102}")