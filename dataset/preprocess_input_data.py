import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import plotly.express as px

train_df    = pd.read_csv("<Path to raw train data>")
test_df     = pd.read_csv("<Path to raw test data>")
combined_df = pd.concat([train_df, test_df], ignore_index=True)

n1 = len(train_df)
n2 = len(test_df)

# label encoding
categorical_cols = ["Gender",
                    "Prior Cardiothoracic Surgery",
                    "Preoperative Chemo - Current Malignancy",
                    "Preoperative Thoracic Radiation Therapy",
                    "Cigarette Smoking",
                    "ECOG Score",
                    "ASA Classification",
                    "Tumor size",
                    "Clinical Staging - Lung Cancer - T",
                    "Clinical Staging - Lung Cancer - N",
                    "Clinical Staging - Lung Cancer - M",
                    "Procedure"]
encoders = dict()
for col in categorical_cols:
    le               = LabelEncoder()
    combined_df[col] = le.fit_transform(combined_df[col])
    encoders[col]    = le

# scaling numerical values
clinical_numerical_cols = ["Age",
        "BMI",
        "FEV1 Predicted",
        "DLCO Predicted",
        "Pack-Years Of Cigarette Use"]
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(train_df[clinical_numerical_cols])
combined_df[clinical_numerical_cols] = scaler.transform(combined_df[clinical_numerical_cols])
if combined_df.isnull().any().any():
    print("There is at least one null value in the DataFrame.")
else:
    print("No null values in the DataFrame.")
combined_df.fillna(0, inplace=True)

with open("<Path to radiomics features json>", "r") as js:
    radiology_feat = json.load(js)['radiomics_features']
scaler = StandardScaler()
scaler.fit(train_df[radiology_feat])
combined_df[radiology_feat] = scaler.transform(combined_df[radiology_feat])

combined_df = combined_df.astype(float)

train_recovered = combined_df.iloc[:n1, :]
test_recovered = combined_df.iloc[n1:, :]

train_recovered.to_csv("<path to save cleaned train data>", index = False)
test_recovered.to_csv("<path to save cleande test data>", index = False)

if combined_df.isnull().any().any():
    print("There is at least one null value in the DataFrame.")
else:
    print("No null values in the DataFrame.")

print(train_recovered.shape)
print(test_recovered.shape)
