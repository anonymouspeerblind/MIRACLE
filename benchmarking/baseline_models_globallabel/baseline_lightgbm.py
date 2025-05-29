import lightgbm as lgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import json
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
X_val  = test_df[clinical_feat]
y_val  = test_df["Target"]

# Create dataset for LightGBM
train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val)

# Define parameter
params = {
    'objective': 'binary',
    'metric': 'binary_error',
    'verbosity': -1,
    'boosting_type': 'gbdt',
    'learning_rate': 0.1,
    'num_leaves': 31,
    'seed': 42
}

# Train the model
model = lgb.train(
    params,
    train_data,
    valid_sets=[train_data, val_data],
    num_boost_round=100
)

# # Predict and evaluate
# y_pred = model.predict(X_val)
# y_pred_binary = (y_pred > 0.5).astype(int)
# accuracy = accuracy_score(y_val, y_pred_binary)
# print(f'Validation Accuracy: {accuracy:.4f}')


y_pred_proba = model.predict(X_val)
fpr, tpr, roc_thresholds   = roc_curve(y_val, y_pred_proba)
auc_value                  = roc_auc_score(y_val, y_pred_proba)
upper_fpr_idx              = min(i for i, val in enumerate(fpr) if val >= 0.2)
tar_at_far_102             = tpr[upper_fpr_idx]
upper_fpr_idx              = min(i for i, val in enumerate(fpr) if val >= 0.3)
tar_at_far_103             = tpr[upper_fpr_idx]

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