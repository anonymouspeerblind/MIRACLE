import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as nn_init
import zero
from huggingface_hub import login
from huggingface_hub import InferenceClient
from transformers import AutoTokenizer, AutoModel
from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator
login(token = "HF token")

class encode_remark(nn.Module):
    def __init__(self):
        super(encode_remark, self).__init__()
        self.model_name = "yikuan8/Clinical-Longformer"
        self.tokenizer  = AutoTokenizer.from_pretrained(self.model_name, cache_dir = "")
        self.model      = AutoModel.from_pretrained(self.model_name, cache_dir = "")
        self.fc1        = nn.Linear(768, 768)
        self.dropout    = nn.Dropout(0.3)
        for params in self.model.parameters():
            params.requires_grad = False

    def forward(self, remarks):
        inputs     = self.tokenizer(remarks, return_tensors="pt", truncation=True, padding='max_length', max_length=512)
        device     = next(self.model.parameters()).device
        inputs     = {key: value.to(device) for key, value in inputs.items()}
        outputs    = self.model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]
        embeddings = self.fc1(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

@variational_estimator
class Clinical_MLP(nn.Module):
    def __init__(self, input_dim=17, hidden_dim=256, embed_dim=768):
        super(Clinical_MLP, self).__init__()
        self.fc1 = nn.Sequential(
            BayesianLinear(input_dim, hidden_dim // 4),
            nn.ReLU(),
            BayesianLinear(hidden_dim // 4, hidden_dim // 2),
            nn.ReLU(),
            BayesianLinear(hidden_dim // 2, hidden_dim),
        )
        self.bn1     = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(0.3)
        self.fc2     = BayesianLinear(hidden_dim, embed_dim)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

@variational_estimator
class Radiology_MLP(nn.Module):
    def __init__(self, input_dim=113, hidden_dim=256, output_dim=768):
        super(Radiology_MLP, self).__init__()
        self.fc1     = BayesianLinear(input_dim, hidden_dim)
        self.bn1     = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(0.3)
        self.fc2     = BayesianLinear(hidden_dim, output_dim)
    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.fc1(x)), 0.1)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

@variational_estimator
class interpretable_model(nn.Module):
    def __init__(self, clinical_encoder, radiology_mlp, remark_encoder, hidden_dim=1024, num_classes=1):
        super().__init__()
        self.mlp1  = clinical_encoder
        self.mlp2  = radiology_mlp
        self.mlp3  = remark_encoder
        concat_dim = 768 * 1
        self.embedding_layer = nn.Sequential(
            BayesianLinear(concat_dim, concat_dim // 2),
            nn.BatchNorm1d(concat_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            BayesianLinear(concat_dim // 2, hidden_dim))
        self.classifier = BayesianLinear(hidden_dim, num_classes)
        self.clinical_weight = 0.5
        self.radiology_weight = 0.25
        self.remark_weight = 0.25

    def forward(self, inp):
        clinical_data, radiology_data, remark = inp
        clinical_embed = self.mlp1(clinical_data)
        radiomic_embed = self.mlp2(radiology_data)
        remark_embed   = self.mlp3(remark)
        
        combined_embed = self.clinical_weight * clinical_embed + self.radiology_weight * radiomic_embed + self.remark_weight * remark_embed
        embedding      = self.embedding_layer(combined_embed)
        logits         = self.classifier(embedding)
        return logits
