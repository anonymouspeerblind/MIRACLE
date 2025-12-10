import pandas as pd
import json

human_df = pd.read_csv("<Path to test data annotated by surgeons>")
test_df = pd.read_csv("<Path to test data>")

with open("<Path to complications list json>", 'r') as js:
    complications = json.load(js)['complications']

total_negatives = (test_df["Target"] == 0).sum()
false_positives = ((test_df["Target"] == 0) & (human_df["SurgeonLabel"] == 1)).sum()
FAR = false_positives / total_negatives

total_positives = (test_df["Target"] == 1).sum()
true_positives = ((test_df["Target"] == 1) & (human_df["SurgeonLabel"] == 1)).sum()
TAR = true_positives / total_positives
print(f"SurgeonLabel: {FAR}")
print(f"SurgeonLabel: {TAR}")
