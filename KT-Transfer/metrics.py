import torch
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, f1_score, confusion_matrix

class MetricCalculator:
    def __init__(self, device='cpu'):
        self.device = device
        self.reset()

    def reset(self):
        """重置存储的预测值和标签"""
        self.probs = []
        self.preds = []
        self.labels = []
        self.losses = []

    def update(self, logits, targets, loss_val=None):
        """
        在每个 batch 结束后调用。
        logits: 模型输出 (Batch, 2) 或 (Batch, 1)
        targets: 真实标签 (Batch,)
        loss_val: 当前 batch 的 loss (scalar)
        """
        # 1. 处理 Loss
        if loss_val is not None:
            self.losses.append(loss_val)

        # 2. 处理 Logits -> Probs & Preds
        with torch.no_grad():
            if logits.shape[1] == 1: # Binary case (e.g. BCEWithLogits)
                prob = torch.sigmoid(logits).view(-1)
                pred = (prob > 0.5).long()
            else: # Multi-class case (e.g. CrossEntropy)
                prob = torch.softmax(logits, dim=1)[:, 1] # 取正类概率
                pred = torch.argmax(logits, dim=1)

            # 3. 存入列表 (转为 CPU numpy)
            self.probs.extend(prob.cpu().numpy())
            self.preds.extend(pred.cpu().numpy())
            self.labels.extend(targets.cpu().numpy())

    def compute(self):
        """
        在 Epoch 结束时调用，计算所有指标。
        返回一个字典。
        """
        y_true = np.array(self.labels)
        y_pred = np.array(self.preds)
        y_prob = np.array(self.probs)

        # 1. 基础指标
        try:
            auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            auc = 0.5 # 防止只有一个类别报错

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0) # Sensitivity

        # 2. 计算 Specificity (特异度)
        # TN / (TN + FP)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        # 3. 平均 Loss
        avg_loss = np.mean(self.losses) if self.losses else 0.0

        return {
            "loss": avg_loss,
            "auc": auc,
            "acc": acc,
            "f1": f1,
            "recall": recall,
            "specificity": specificity
        }
