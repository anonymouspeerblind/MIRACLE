import os
import numpy as np
from tqdm import tqdm
import json
import argparse
import torch
import torch.nn as nn
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from sklearn.metrics import roc_curve, roc_auc_score
import plotly.express as px
from transformers import get_cosine_schedule_with_warmup

from model import interpretable_model, encode_remark, Clinical_MLP, Radiology_MLP
from loss import BCE_normal_loss, BCE_typical_weighted_loss, BCE_weighted_loss, BCE_smoothed_weighted_loss, AsymmetricLossOptimized, CE_Weighted, FocalLoss
from dataloader import Global_label
from torch.utils.tensorboard import SummaryWriter

def custom_collate_fn(batch):
    clinical_batch  = torch.stack([item[0] for item in batch], dim=0)
    radiology_batch = torch.stack([item[1] for item in batch], dim=0)
    target_batch    = torch.stack([item[2] for item in batch], dim=0)
    remarks_batch   = [item[3] for item in batch]
    return clinical_batch, radiology_batch, target_batch, remarks_batch

def train(args, model, train_loader, device, optimizer, loss_fn):
    model.train()
    steploss = list()
    for batch_idx, (clinical_data, radiology_data, target_data, remark) in enumerate(pbar := tqdm(train_loader)):
        clinical_data, radiology_data, target_data = clinical_data.to(device), radiology_data.to(device), target_data.to(device)
        optimizer.zero_grad()
        
        loss = model.sample_elbo(
            (clinical_data, radiology_data, remark), target_data, 
            criterion=loss_fn, 
            sample_nbr=10,
            complexity_cost_weight=0.000001
        )
        loss.backward()
        optimizer.step()
        steploss.append(loss)
        pbar.set_description(f"Loss {loss}")
    return sum(steploss)/len(steploss)

def test(model, test_loader, device, epoch, plot_dir, model_name, lr, mc_samples=10):
    model.eval()
    all_targets, all_probs = list(), list()
    with torch.no_grad():
        for batch_idx, (clinical_data, radiology_data, target_data, remark) in enumerate(pbar := tqdm(test_loader)):
            clinical_data  = clinical_data.to(device)
            radiology_data = radiology_data.to(device)

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

    binary_predictions       = (all_probs >= optimal_threshold).astype(int).tolist()
    binary_predictions       = np.array(binary_predictions)
    targets                  = np.array(all_targets)

    return tar_at_far_102, tar_at_far_103, auc_value, binary_predictions

def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-r', '--remarks', type=str, default='', help='remarks file')
    parser.add_argument('-hum', '--human', action='store_true', default=False)
    parser.add_argument('-inp', '--input_type', type=str, help='input type')
    parser.add_argument('-bs', '--batch_size', type=int, default=32, help='batch size for training')
    parser.add_argument('-tbs', '--test_batch_size', type=int, default=8, help='batch size for testing')
    parser.add_argument('-e', '--epochs', type=int, default=50, help='Epochs')
    parser.add_argument('-lr', '--lr', type=float, default=1.0, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.9, help='Learning rate step gamma')
    parser.add_argument('--model_name', type=str, default=str(), help='model name')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    args   = parser.parse_args()

    log_writer = SummaryWriter("",comment = None)

    if not os.path.exists(""):   # creating checkpoints folder
        os.mkdir("")
    if not os.path.exists(""):   # creating plots folder
        os.mkdir("")

    plot_dir             = ""
    checkpoint_save_path = ""
    use_cuda             = not args.no_cuda and torch.cuda.is_available()
    device               = torch.device("cuda" if use_cuda else "cpu")
    train_kwargs         = {'batch_size': args.batch_size, 'shuffle': True}
    test_kwargs          = {'batch_size': args.test_batch_size, 'shuffle': False}
    if use_cuda:
        cuda_kwargs = {
                       'num_workers': 4,
                       'pin_memory': True
                       }
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    train_dataset = Global_label(split = "train", remarks_file = args.remarks, human = args.human)
    val_dataset   = Global_label(split = "val", remarks_file = args.remarks, human = args.human)
    test_dataset  = Global_label(split = "test", remarks_file = args.remarks, human = args.human)
    train_loader  = torch.utils.data.DataLoader(train_dataset, **train_kwargs, collate_fn = custom_collate_fn)
    val_loader    = torch.utils.data.DataLoader(val_dataset, **test_kwargs, collate_fn = custom_collate_fn)
    test_loader   = torch.utils.data.DataLoader(test_dataset, **test_kwargs, collate_fn = custom_collate_fn)

    model     = interpretable_model(Clinical_MLP(), Radiology_MLP(), encode_remark(), hidden_dim=256, num_classes=1)
    model.to(device)
    loss_fn   = FocalLoss(gamma=4, alpha=0.8, reduction='mean', task_type='binary', num_classes=None)
    optimizer = AdamW([{'params': list(model.mlp1.parameters()) + list(model.mlp2.parameters()) + list(model.mlp3.parameters()) + list(model.embedding_layer.parameters()) + list(model.classifier.parameters()),'lr': args.lr}], weight_decay=0.0001)
    scheduler = StepLR(optimizer, step_size =  4, gamma = args.gamma)

    roc_lst         = list()
    tar_far_103_lst = list()
    tar_far_102_lst = list()
    print("For Epoch 0.................")
    tar_at_far_102, tar_at_far_103, auc_value, binary_predictions = test(model, val_loader, device, 0, plot_dir, args.model_name, args.lr)
    log_writer.add_scalar('AUC/epoch',auc_value,0)
    log_writer.add_scalar('LR/epoch',scheduler.get_last_lr()[0],0)
    log_writer.add_scalar('TAR@FAR0.2/epoch',tar_at_far_102,0)
    log_writer.add_scalar('TAR@FAR0.3/epoch',tar_at_far_103,0)

    for epoch in range(1, args.epochs + 1):
        print(f"running epoch------ {epoch}")
        avg_step_loss = train(args, model, train_loader, device, optimizer, loss_fn)
        
        # evaluation
        tar_at_far_102, tar_at_far_103, auc_value, binary_predictions = test(model, val_loader, device, epoch, plot_dir, args.model_name, args.lr)

        print(f"AUC for {epoch}: {auc_value}")
        print(f"TAR@FAR=0.3 for {epoch}: {tar_at_far_103}")
        print(f"TAR@FAR=0.2 for {epoch}: {tar_at_far_102}")

        roc_lst.append(auc_value)
        tar_far_103_lst.append(tar_at_far_103)
        tar_far_102_lst.append(tar_at_far_102)

        log_writer.add_scalar('AUC/epoch',auc_value,epoch)
        log_writer.add_scalar('LR/epoch',scheduler.get_last_lr()[0],epoch)
        log_writer.add_scalar('TAR@FAR0.2/epoch',tar_at_far_102,epoch)
        log_writer.add_scalar('TAR@FAR0.3/epoch',tar_at_far_103,epoch)
        log_writer.add_scalar('AvgLoss/epoch',avg_step_loss,epoch)

        torch.save(model.state_dict(), checkpoint_save_path + args.input_type + "_" + args.model_name + "_lr_" + str(args.lr) + "_bs_" + str(args.batch_size) + "_" + str(epoch) + "_" + str(auc_value) + "_" + str(tar_at_far_103) + "_" + str(tar_at_far_102) + ".pt")
    log_writer.close()

    print(f"Highest AUC at epoch {roc_lst.index(max(roc_lst)) + 1} epoch with ROC = {max(roc_lst)}")
    print(f"Highest TAR@FAR=0.3 at epoch {tar_far_103_lst.index(max(tar_far_103_lst)) + 1} epoch with TAR = {max(tar_far_103_lst)}")
    print(f"Highest TAR@FAR=0.2 at epoch {tar_far_102_lst.index(max(tar_far_102_lst)) + 1} epoch with TAR = {max(tar_far_102_lst)}")

if __name__ == '__main__':
    main()