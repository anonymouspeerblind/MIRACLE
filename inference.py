import os
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
import json
import argparse
import plotly.express as px
import torch
import torch.nn as nn
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from sklearn.metrics import roc_curve, roc_auc_score
from dataloader import Global_label
from model import interpretable_model, encode_remark, Clinical_MLP, Radiology_MLP

def custom_collate_fn(batch):
    clinical_batch  = torch.stack([item[0] for item in batch], dim=0)
    radiology_batch = torch.stack([item[1] for item in batch], dim=0)
    target_batch    = torch.stack([item[2] for item in batch], dim=0)
    remarks_batch   = [item[3] for item in batch]
    return clinical_batch, radiology_batch, target_batch, remarks_batch

def test(model, test_loader, device, rem, mc_samples=10):
    model.eval()
    all_targets, all_probs = list(), list()
    with torch.no_grad():
        for batch_idx, (clinical_data, radiology_data, target_data, remark) in enumerate(pbar := tqdm(test_loader)):
            clinical_data  = clinical_data.to(device)
            radiology_data = radiology_data.to(device)

            # Monte-Carlo averaging
            mc_outputs = list()
            for _ in range(mc_samples):
                logits = model((clinical_data, radiology_data, remark))
                probs  = torch.sigmoid(logits)
                mc_outputs.append(probs)

            avg_probs = torch.mean(torch.stack(mc_outputs), dim=0)

            all_probs.append(avg_probs.cpu().numpy())
            all_targets.append(target_data.cpu().numpy())
    all_probs    = np.concatenate(all_probs, axis=0)
    all_targets  = np.concatenate(all_targets, axis=0)

    fpr, tpr, roc_thresholds = roc_curve(all_targets, all_probs)

    youden_index             = tpr - fpr
    max_index                = np.argmax(youden_index)
    optimal_threshold        = roc_thresholds[max_index]
    auc_value                = roc_auc_score(all_targets, all_probs)
    upper_fpr_idx            = min(i for i, val in enumerate(fpr) if val >= 0.3)
    tar_at_far_103           = tpr[upper_fpr_idx]
    upper_fpr_idx            = min(i for i, val in enumerate(fpr) if val >= 0.2)
    tar_at_far_102           = tpr[upper_fpr_idx]
    np.savez("final_model_roc_values_"+rem+".npz",
             fpr=fpr,
             tpr=tpr,
             thresholds=roc_thresholds,
             auc=auc_value,
             tar30=tar_at_far_103,
             tar20=tar_at_far_102)

    return tar_at_far_102, tar_at_far_103, auc_value

if __name__ == "__main__":
    device = torch.device("cuda")
    # model  = interpretable_model(Clinical_MLP())                                                                      # only clinical model
    # model  = interpretable_model(Clinical_MLP(), Radiology_MLP())                                                     # clinical + radiological model
    model  = interpretable_model(Clinical_MLP(), Radiology_MLP(), encode_remark(), hidden_dim=256, num_classes=1)       # entire model
    ckpt   = torch.load("<Path to trained model checkpoint>", map_location=torch.device('cpu'))
    model.load_state_dict(ckpt,strict = True)
    model.to(device)

    remarks_type = "<remarks type>"
    
    test_dataset = Global_label(split = "test", remarks_file = remarks_type, human = False)
    test_loader  = torch.utils.data.DataLoader(test_dataset, shuffle = False, batch_size = 1, num_workers = 4, pin_memory = True, collate_fn = custom_collate_fn)
    tar_at_far_102, tar_at_far_103, auc_value, binary_predictions = test(model, test_loader, device, "saving remark for npy matrix")

    print(f"AUCROC: {auc_value}")
    print(f"TAR@FAR=0.2: {tar_at_far_102}")
    print(f"TAR@FAR=0.3: {tar_at_far_103}")