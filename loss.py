import json
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.init
import torch.nn.functional as F

def apply_label_smoothing(targets, smoothing_factor=0.1):
    smoothed_targets = targets * (1 - smoothing_factor) + 0.5 * smoothing_factor
    return smoothed_targets

class MS_CE_Loss(nn.Module):
    def __init__(self, margin=0, max_violation=False):
        super(MS_CE_Loss, self).__init__()
        self.margin = margin
        self.max_violation = max_violation
        self.thresh = 0.2  # 0.2
        self.margin = 0.7  #0.5, 0.7
        self.scale_pos = 4.0  #4.0
        self.scale_neg = 20.0  #20.0
        self.loss_fn = nn.CrossEntropyLoss(weight=torch.Tensor([5/32, 27/32]).cuda())
        self.epsilon = 1e-5

    def l2_norm(self, input):
        input_size = input.size()
        buffer  = torch.pow(input, 2)
        normp   = torch.sum(buffer, 1).add_(1e-12)
        norm    = torch.sqrt(normp)
        _output = torch.div(input, norm.view(-1, 1).expand_as(input))
        output  = _output.view(input_size)
        return output

    def ms_sample(self, sim_mat, label):
        epsilon_mask = (sim_mat < 1 - self.epsilon).float()
        pos_mask = ((label.view(-1, 1) == label.view(1, -1)).float().cuda() * epsilon_mask)
        diag_mask = 1 - torch.eye(sim_mat.size(0), device=sim_mat.device)
        pos_mask = pos_mask * diag_mask
        neg_mask = (1 - pos_mask) * diag_mask  # now excludes diagonal

        pos_exp = torch.exp(-self.scale_pos * (sim_mat - self.thresh))
        neg_exp = torch.exp(self.scale_neg * (sim_mat - self.thresh))

        P_sim = torch.where(pos_mask == 1, sim_mat, torch.ones_like(pos_exp) * 1e16)
        N_sim = torch.where(neg_mask == 1, sim_mat, torch.ones_like(neg_exp) * -1e16)

        min_P_sim, _ = torch.min(P_sim, dim=1, keepdim=True)
        max_N_sim, _ = torch.max(N_sim, dim=1, keepdim=True)

        hard_P_sim = torch.where(P_sim - self.margin < max_N_sim, pos_exp, torch.zeros_like(pos_exp)).sum(dim=-1)
        hard_N_sim = torch.where(N_sim + self.margin > min_P_sim, neg_exp, torch.zeros_like(neg_exp)).sum(dim=-1)

        pos_loss = torch.log(1 + hard_P_sim).sum() / self.scale_pos
        neg_loss = torch.log(1 + hard_N_sim).sum() / self.scale_neg
        print(f"pos_loss and neg_loss ------------------- {pos_loss, neg_loss}")

        return (4*pos_loss) + neg_loss

    def forward(self, embeddings, logits, labels):
        sim_mat = F.linear(self.l2_norm(embeddings), self.l2_norm(embeddings))
        ms_loss = self.ms_sample(sim_mat,labels)
        ce_loss = self.loss_fn(logits, labels.long())
        print(f"Losses ----------------- {ms_loss}")
        return (2*ce_loss) + ms_loss / 100.0

class BCE_typical_weighted_loss(nn.Module):
    def __init__(self, pos_weight_value=27/5, device='cuda'):
        """
        Initialize BCEWithLogitsLoss with a positive class weight to handle skewed class ratios.
        
        Args:
            pos_weight_value (float): The weight to assign to the positive class.
            device (str): The device on which the pos_weight tensor should be allocated.
        """
        super(BCE_typical_weighted_loss, self).__init__()
        # Create the pos_weight tensor with a single element for binary classification
        self.pos_weight = torch.tensor([pos_weight_value]).to(device)
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

    def forward(self, prediction, target_data):
        """
        Compute the weighted BCEWithLogits loss.

        Args:
            prediction (torch.Tensor): The raw logits output by the model.
            target_data (torch.Tensor): The ground truth binary labels.
            
        Returns:
            torch.Tensor: The computed loss.
        """
        prediction = prediction.squeeze(dim=-1)
        print(sum(target_data), len(target_data))
        return self.loss_fn(prediction, target_data)

class CE_Weighted(nn.Module):
    def __init__(self):
        super(CE_Weighted, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss(weight=torch.Tensor([5/32, 27/32]).cuda())
    def forward(self, prediction, target_data):
        print(sum(target_data), len(target_data))
        return self.loss_fn(prediction, target_data.long())

class BCE_normal_loss(nn.Module):
    def __init__(self):
        super(BCE_normal_loss, self).__init__()
        self.loss_fn = nn.BCEWithLogitsLoss()
    def forward(self, prediction, target_data):
        return self.loss_fn(prediction, target_data)

class BCE_weighted_loss(nn.Module):
    def __init__(self):
        super(BCE_weighted_loss, self).__init__()
        with open("/home/spandey8/roswell_park/Postoperative_complications/miccai_paper/dl_codebase/dataset/complications_list.json", "r") as js:
            self.complications  = json.load(js)['complications']
        self.df                 = pd.read_csv("/home/spandey8/roswell_park/Postoperative_complications/miccai_paper/dl_codebase/dataset/train_cleaned_input.csv")
        self.weights            = list()
        for comp in self.complications:
            self.weights.append(self.df.shape[0]/(self.df[comp].value_counts().get(1, 0) * len(self.complications)))
        self.pos_weigh = torch.tensor(self.weights).to('cuda')
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight = self.pos_weigh)
    def forward(self, prediction, target_data):
        return self.loss_fn(prediction, target_data)

class BCE_smoothed_weighted_loss(nn.Module):
    def __init__(self, smoothing_factor = 0.1):
        super(BCE_smoothed_weighted_loss, self).__init__()
        self.smoothing_factor = smoothing_factor
        with open("/home/spandey8/roswell_park/Postoperative_complications/miccai_paper/dl_codebase/dataset/complications_list.json", "r") as js:
            self.complications  = json.load(js)['complications']
        self.df                 = pd.read_csv("/home/spandey8/roswell_park/Postoperative_complications/miccai_paper/dl_codebase/dataset/train_cleaned_input.csv")
        self.weights            = list()
        for comp in self.complications:
            self.weights.append(self.df.shape[0]/(self.df[comp].value_counts().get(1, 0) * len(self.complications)))
        self.pos_weigh = torch.tensor(self.weights).to('cuda')
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight = self.pos_weigh)
    def forward(self, prediction, target_data):
        smoothed_targets = target_data * (1 - self.smoothing_factor) + 0.5 * self.smoothing_factor
        return self.loss_fn(prediction, smoothed_targets)

class AsymmetricLossOptimized(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """
        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                          self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            self.loss *= self.asymmetric_w

        return -self.loss.sum()

class FocalLoss(nn.Module):
    def __init__(self, gamma=4, alpha=None, reduction='mean', task_type='binary', num_classes=None):
        """
        Unified Focal Loss class for binary, multi-class, and multi-label classification tasks.
        :param gamma: Focusing parameter, controls the strength of the modulating factor (1 - p_t)^gamma
        :param alpha: Balancing factor, can be a scalar or a tensor for class-wise weights. If None, no class balancing is used.
        :param reduction: Specifies the reduction method: 'none' | 'mean' | 'sum'
        :param task_type: Specifies the type of task: 'binary', 'multi-class', or 'multi-label'
        :param num_classes: Number of classes (only required for multi-class classification)
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

        self.reduction = reduction
        self.task_type = task_type
        self.num_classes = num_classes

        # Handle alpha for class balancing in multi-class tasks
        if task_type == 'multi-class' and alpha is not None and isinstance(alpha, (list, torch.Tensor)):
            assert num_classes is not None, "num_classes must be specified for multi-class classification"
            if isinstance(alpha, list):
                self.alpha = torch.Tensor(alpha)
            else:
                self.alpha = alpha

    def forward(self, inputs, targets):
        """
        Forward pass to compute the Focal Loss based on the specified task type.
        :param inputs: Predictions (logits) from the model.
                       Shape:
                         - binary/multi-label: (batch_size, num_classes)
                         - multi-class: (batch_size, num_classes)
        :param targets: Ground truth labels.
                        Shape:
                         - binary: (batch_size,)
                         - multi-label: (batch_size, num_classes)
                         - multi-class: (batch_size,)
        """
        if self.task_type == 'binary':
            return self.binary_focal_loss(inputs, targets)
        elif self.task_type == 'multi-class':
            return self.multi_class_focal_loss(inputs, targets)
        elif self.task_type == 'multi-label':
            return self.multi_label_focal_loss(inputs, targets)
        else:
            raise ValueError(
                f"Unsupported task_type '{self.task_type}'. Use 'binary', 'multi-class', or 'multi-label'.")

    def binary_focal_loss(self, inputs, targets):
        """ Focal loss for binary classification. """
        if targets.dim() < inputs.dim():
            targets = targets.unsqueeze(1)
        probs = torch.sigmoid(inputs)
        targets = targets.float()
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            bce_loss = alpha_t * bce_loss

        loss = focal_weight * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

    def multi_class_focal_loss(self, inputs, targets):
        """ Focal loss for multi-class classification. """
        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)

        # Convert logits to probabilities with softmax
        probs = F.softmax(inputs, dim=1)

        # One-hot encode the targets
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).float()

        # Compute cross-entropy for each class
        ce_loss = -targets_one_hot * torch.log(probs)

        # Compute focal weight
        p_t = torch.sum(probs * targets_one_hot, dim=1)  # p_t for each sample
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha if provided (per-class weighting)
        if self.alpha is not None:
            alpha_t = alpha.gather(0, targets)
            ce_loss = alpha_t.unsqueeze(1) * ce_loss

        # Apply focal loss weight
        loss = focal_weight.unsqueeze(1) * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

    def multi_label_focal_loss(self, inputs, targets):
        """ Focal loss for multi-label classification. """
        probs = torch.sigmoid(inputs)
        # Compute binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        # Compute focal weight
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha if provided
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            bce_loss = alpha_t * bce_loss

        # Apply focal loss weight
        loss = focal_weight * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss