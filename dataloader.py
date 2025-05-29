import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import json
from tqdm import tqdm

class Global_label(Dataset):
    def __init__(self, split, remarks_file, human = True):
        self.split            = split
        self.remarks_file     = remarks_file
        self.human            = human
        with open("<Path for clinical features list file>", "r") as js:
            self.clinical_features = json.load(js)['clinical_features']
        with open("<Path for complications list file>", "r") as js:
            self.complications_lst = json.load(js)['complications']
        if self.split == "train":
            self.clinical_data     = pd.read_csv("<Training split file>")
            self.record_ids        = self.clinical_data['Record'].values.tolist()
            self.clinical_data_inp = self.clinical_data[self.clinical_features].values.tolist()
            self.target            = self.clinical_data['Target'].values.tolist()
            self.radio_data_inp    = self.clinical_data[[col for col in self.clinical_data if col not in self.clinical_features and col not in self.complications_lst and col not in ['Record', 'Target']]].values.tolist()
            with open("<Path to train split remarks file>", "r") as js:
                self.remarks       = list(json.load(js).items())
            with open("<Path to train split remarks file>", "r") as js:
                self.remarks_dict  = json.load(js)
        elif self.split == "val":
            self.clinical_data     = pd.read_csv("<Validation split file>")
            self.record_ids        = self.clinical_data['Record'].values.tolist()
            self.clinical_data_inp = self.clinical_data[self.clinical_features].values.tolist()
            self.target            = self.clinical_data['Target'].values.tolist()
            self.radio_data_inp    = self.clinical_data[[col for col in self.clinical_data if col not in self.clinical_features and col not in self.complications_lst and col not in ['Record', 'Target']]].values.tolist()
            with open("<Path to validation split remarks file>", "r") as js:
                self.remarks       = list(json.load(js).items())
            with open("<Path to validation split remarks file>", "r") as js:
                self.remarks_dict  = json.load(js)
        else:
            self.clinical_data     = pd.read_csv("<Testing split file>")
            self.record_ids        = self.clinical_data['Record'].values.tolist()
            self.clinical_data_inp = self.clinical_data[self.clinical_features].values.tolist()
            self.target            = self.clinical_data['Target'].values.tolist()
            self.radio_data_inp    = self.clinical_data[[col for col in self.clinical_data if col not in self.clinical_features and col not in self.complications_lst and col not in ['Record', 'Target']]].values.tolist()
            if self.human:
                self.human_remark      = pd.read_csv("<Path to surgeon annotation file>")
                self.remark_dic        = dict(zip(self.human_remark["Record"], self.human_remark["Remarks"]))
            else:
                with open("<Path to testing split remarks file>", "r") as js:
                    self.remarks       = list(json.load(js).items())
                with open("<Path to testing split remarks file>", "r") as js:
                    self.remarks_dict  = json.load(js)
    def __len__(self):
        return self.clinical_data.shape[0]
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.split == 'train':
            clinical_data  = torch.tensor(self.clinical_data_inp[idx], dtype=torch.float32)
            radiology_data = torch.tensor(self.radio_data_inp[idx], dtype=torch.float32)
            target_data    = torch.tensor(self.target[idx], dtype=torch.float32)
            record_id      = self.record_ids[idx]
            remark         = self.remarks_dict[str(record_id)]['remarks']
            return clinical_data, radiology_data, target_data, remark
        elif self.split == 'val':
            clinical_data  = torch.tensor(self.clinical_data_inp[idx], dtype=torch.float32)
            radiology_data = torch.tensor(self.radio_data_inp[idx], dtype=torch.float32)
            target_data    = torch.tensor(self.target[idx], dtype=torch.float32)
            record_id      = self.record_ids[idx]
            remark         = self.remarks_dict[str(record_id)]['remarks']
            return clinical_data, radiology_data, target_data, remark
        else:
            clinical_data  = torch.tensor(self.clinical_data_inp[idx], dtype=torch.float32)
            radiology_data = torch.tensor(self.radio_data_inp[idx], dtype=torch.float32)
            target_data    = torch.tensor(self.target[idx], dtype=torch.float32)
            if self.human:
                record     = self.record_ids[idx]
                remark     = self.remark_dic[record]
            else:
                record  = self.record_ids[idx]
                remark  = self.remarks_dict[str(record)]['remarks']
            return clinical_data, radiology_data, target_data, remark