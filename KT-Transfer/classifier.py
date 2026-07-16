import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import torch
import torchmetrics
import wandb
from matplotlib import pyplot as plt

# 假设 utils 在同级目录
from utils import get_loss, get_model, get_optimizer, get_scheduler

class ClassifierLightning(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 1. 初始化模型
        self.model = get_model(
            self.config.model,
            num_classes=self.config.num_classes,
            input_dim=config.input_dim,
            **self.config.model_config,
        )
        
        # 2. 初始化损失函数
        self.criterion = get_loss(config.criterion, pos_weight=config.pos_weight
                                 ) if config.task == "binary" else get_loss(config.criterion)
        
        self.save_hyperparameters()
        self.lr = config.lr
        self.wd = config.wd

        # 3. 初始化指标
        # torchmetrics.Accuracy 默认 threshold=0.5 (Binary)
        self.acc_train = torchmetrics.Accuracy(task=config.task, num_classes=config.num_classes)
        self.acc_val = torchmetrics.Accuracy(task=config.task, num_classes=config.num_classes)
        self.acc_test = torchmetrics.Accuracy(task=config.task, num_classes=config.num_classes)

        self.auroc_val = torchmetrics.AUROC(task=config.task, num_classes=config.num_classes)
        self.auroc_test = torchmetrics.AUROC(task=config.task, num_classes=config.num_classes)

        self.f1_val = torchmetrics.F1Score(task=config.task, num_classes=config.num_classes)
        self.f1_test = torchmetrics.F1Score(task=config.task, num_classes=config.num_classes)

        self.precision_val = torchmetrics.Precision(task=config.task, num_classes=config.num_classes)
        self.precision_test = torchmetrics.Precision(task=config.task, num_classes=config.num_classes)

        self.recall_val = torchmetrics.Recall(task=config.task, num_classes=config.num_classes)
        self.recall_test = torchmetrics.Recall(task=config.task, num_classes=config.num_classes)

        self.specificity_val = torchmetrics.Specificity(task=config.task, num_classes=config.num_classes)
        self.specificity_test = torchmetrics.Specificity(task=config.task, num_classes=config.num_classes)

        self.cm_val = torchmetrics.ConfusionMatrix(task=config.task, num_classes=config.num_classes)
        self.cm_test = torchmetrics.ConfusionMatrix(task=config.task, num_classes=config.num_classes)

    def forward(self, x, *args):
        logits = self.model(x, *args)
        return logits

    def configure_optimizers(self):
        optimizer = get_optimizer(
            name=self.config.optimizer,
            model=self.model,
            lr=self.lr,
            wd=self.wd,
        )
        if self.config.lr_scheduler:
            scheduler = get_scheduler(
                self.config.lr_scheduler,
                optimizer,
                **self.config.lr_scheduler_config,
            )
            return [optimizer], [scheduler]
        else:
            return [optimizer]

    # ================= Training =================
    def training_step(self, batch, batch_idx):
        # 根据你的 Dataset 返回值解包 (x, coords, y, ...)
        x, coords, y, _, _ = batch 
        
        logits = self.forward(x, coords)
        
        if self.config.task == "binary":
            loss = self.criterion(logits, y.unsqueeze(1).float())
            # 计算概率用于 metric (Accuracy 会自动处理阈值)
            preds_or_probs = torch.sigmoid(logits)
        else:
            loss = self.criterion(logits, y)
            preds_or_probs = torch.softmax(logits, dim=1)

        # 记录 Train 指标
        self.acc_train(preds_or_probs, y)
        self.log("acc/train", self.acc_train, on_step=True, on_epoch=True, prog_bar=True)
        self.log("loss/train", loss, on_step=True, on_epoch=True, prog_bar=False)

        return loss

    # ================= Validation =================
    def validation_step(self, batch, batch_idx):
        x, coords, y, _, _ = batch
        logits = self.forward(x, coords)
        
        if self.config.task == "binary":
            loss = self.criterion(logits, y.unsqueeze(1).float())
            probs = torch.sigmoid(logits)
        else:
            loss = self.criterion(logits, y)
            probs = torch.softmax(logits, dim=1)

        # 更新所有 Val 指标
        self.acc_val(probs, y)
        self.auroc_val(probs, y)
        self.f1_val(probs, y)
        self.precision_val(probs, y)
        self.recall_val(probs, y)
        self.specificity_val(probs, y)
        self.cm_val(probs, y)

        # Log
        self.log("loss/val", loss, prog_bar=True)
        self.log("acc/val", self.acc_val, on_epoch=True, prog_bar=True)
        self.log("auroc/val", self.auroc_val, on_epoch=True, prog_bar=True)
        self.log("f1/val", self.f1_val, on_epoch=True)
        self.log("precision/val", self.precision_val, on_epoch=True)
        self.log("recall/val", self.recall_val, on_epoch=True)
        self.log("specificity/val", self.specificity_val, on_epoch=True)

        return loss

    def on_validation_epoch_end(self):
        # 绘制并记录混淆矩阵
        cm = self.cm_val.compute()
        self.log_confusion_matrix(cm, "val")
        self.cm_val.reset()

    # ================= Test =================
    def on_test_epoch_start(self):
        # 初始化 DataFrame 存储测试结果
        self.outputs = pd.DataFrame(columns=['patient', 'ground_truth', 'prediction', 'logits', 'correct'])

    def test_step(self, batch, batch_idx):
        x, coords, y, _, patient = batch
        logits = self.forward(x, coords)

        if self.config.task == "binary":
            loss = self.criterion(logits, y.unsqueeze(1).float())
            probs = torch.sigmoid(logits)
            # 默认 0.5 阈值生成预测标签
            preds = torch.round(probs) 
        else:
            loss = self.criterion(logits, y)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

        # 更新所有 Test 指标
        self.acc_test(probs, y)
        self.auroc_test(probs, y)
        self.f1_test(probs, y)
        self.precision_test(probs, y)
        self.recall_test(probs, y)
        self.specificity_test(probs, y)
        self.cm_test(probs, y)

        # Log
        self.log("loss/test", loss)
        self.log("acc/test", self.acc_test, on_epoch=True)
        self.log("auroc/test", self.auroc_test, on_epoch=True)
        self.log("f1/test", self.f1_test, on_epoch=True)
        self.log("precision/test", self.precision_test, on_epoch=True)
        self.log("recall/test", self.recall_test, on_epoch=True)
        self.log("specificity/test", self.specificity_test, on_epoch=True)

        # 保存详细结果到 DataFrame
        # 注意：这里需要把 tensor 转回 cpu numpy
        batch_outputs = pd.DataFrame({
            'patient': patient,
            'ground_truth': y.cpu().numpy(),
            'prediction': preds.squeeze().cpu().numpy(),
            # 保存 logits 或 probs 都可以，方便后续分析
            'logits': logits.squeeze().detach().cpu().numpy(), 
            'correct': (y == preds.squeeze()).int().cpu().numpy()
        })
        self.outputs = pd.concat([self.outputs, batch_outputs], ignore_index=True)

    def on_test_epoch_end(self):
        # 绘制并记录混淆矩阵
        cm = self.cm_test.compute()
        self.log_confusion_matrix(cm, "test")
        self.cm_test.reset()
        
        # 打印一下最终结果
        print("\nTest Results (Default Threshold 0.5):")
        # 注意：这里获取的是整个 epoch 累积计算后的值
        print(f"ACC: {self.acc_test.compute():.4f}")
        print(f"AUC: {self.auroc_test.compute():.4f}")

    # --- 辅助函数：画混淆矩阵 ---
    def log_confusion_matrix(self, cm, stage):
        # 归一化 (按行，即按真实标签)
        norm = cm.sum(axis=1, keepdims=True)
        # 防止除以0
        norm = torch.where(norm == 0, torch.tensor(1).to(norm), norm)
        normalized_cm = cm / norm
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(normalized_cm.cpu(), annot=cm.cpu(), cmap='rocket_r', fmt='g')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix ({stage})')
        
        # 如果使用了 WandB Logger，则记录图片
        if isinstance(self.logger, pl.loggers.WandbLogger):
            wandb.log({f"confusion_matrix/{stage}": wandb.Image(plt)})
        plt.close()
