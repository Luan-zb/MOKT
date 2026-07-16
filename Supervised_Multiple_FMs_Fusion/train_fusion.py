import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
import numpy as np
from tqdm import tqdm
from scipy.stats import pearsonr
import json

from fusion_architectures import MultiModelFusion
from fusion_dataset import FusionDataset, collate_fusion

# ==========================================
# 1. Loss Functions
# ==========================================
class CorrelationLoss(nn.Module):
    """皮尔逊相关系数 Loss: 1 - Correlation"""
    def __init__(self):
        super(CorrelationLoss, self).__init__()

    def forward(self, y_pred, y_true):
        y_pred_mean = y_pred.mean(dim=1, keepdim=True)
        y_true_mean = y_true.mean(dim=1, keepdim=True)
        
        y_pred_std = y_pred - y_pred_mean
        y_true_std = y_true - y_true_mean
        
        corr = (y_pred_std * y_true_std).sum(dim=1) / (
            torch.sqrt((y_pred_std**2).sum(dim=1) + 1e-8) * 
            torch.sqrt((y_true_std**2).sum(dim=1) + 1e-8) + 1e-8
        )
        return 1 - corr.mean()

def compute_pearson(y_pred, y_true):
    """计算 Batch 的平均皮尔逊相关系数"""
    p_corrs = []
    for i in range(y_pred.shape[0]):
        r, _ = pearsonr(y_pred[i], y_true[i])
        if not np.isnan(r):
            p_corrs.append(r)
    return np.mean(p_corrs) if p_corrs else 0.0

# ==========================================
# 2. Train / Eval Logic
# ==========================================
def run_epoch(model, loader, criterion_mse, criterion_corr, optimizer, device, is_train=True):
    model.train() if is_train else model.eval()
    total_mse = 0
    total_corr = 0
    all_pearson = []
    
    pbar = tqdm(loader, desc="Training" if is_train else "Evaluating", leave=False)
    
    with torch.set_grad_enabled(is_train):
        for m_batches, labels, _ in pbar:
            m_batches = [m.to(device) for m in m_batches]
            labels = labels.to(device)
            
            outputs, _ = model(m_batches)
            
            mse_loss = criterion_mse(outputs, labels)
            corr_loss = criterion_corr(outputs, labels)
            loss = mse_loss + 0.5 * corr_loss
            
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            total_mse += mse_loss.item()
            total_corr += corr_loss.item()
            
            batch_p = compute_pearson(outputs.detach().cpu().numpy(), labels.detach().cpu().numpy())
            all_pearson.append(batch_p)
            
            pbar.set_postfix({"MSE": f"{mse_loss.item():.4f}", "P": f"{batch_p:.4f}"})
            
    n = len(loader)
    return total_mse / n, total_corr / n, np.mean(all_pearson)

def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Model Fusion 5-Fold CV with 3:1:1 Split")
    parser.add_argument('--data_dir', type=str, default="/data/ruiyan/lhj/TRIDENT/BRCA_LUAD_LUSC_STAD_LIHC_BLCA_BALANCED_WSI_FEATURES")
    parser.add_argument('--gene_csv', type=str, default="/data/ruiyan/lhj/gene_expression_label/combined_all_gene_expression_matrix.csv")
    parser.add_argument('--save_dir', type=str, default="/data/ruiyan/lhj/models_fusion/logs/alldata_fms_checkpoints")
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--n_folds', type=int, default=5, help='Number of folds for cross-validation')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()

# ==========================================
# 3. Main Loop (5-Fold CV with 3:1:1 Split)
# ==========================================
def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # input_dims = [768, 512, 512, 1280,768] # CHIEF, Feather, madeleine, prism,titan
    # model_names = ['CHIEF', 'Feather', 'madeleine', 'prism','titan']
    input_dims = [768, 512, 512, 1280,768, 768] # CHIEF, Feather, madeleine, prism,titan
    model_names = ['CHIEF', 'Feather', 'madeleine', 'prism','titan','GigaPath']  
    print("="*60)
    print("Multi-Model Fusion 5-Fold CV with 3:1:1 Split")
    print("="*60)
    print(f"每折划分: Train (60%) / Val (20%) / Test (20%)")
    
    # 1. 加载完整数据集
    print("\n[1] 加载数据集...")
    full_dataset = FusionDataset(args.data_dir, args.gene_csv, model_names=model_names)
    gene_dim = full_dataset.gene_dim
    total_samples = len(full_dataset)
    
    # 2. 5-Fold Cross Validation with Patient-Level Grouping (防止数据泄露)
    # 使用 GroupKFold 确保同一患者的所有切片在同一折中
    print(f"\n[2] 5-Fold 交叉验证划分 (按患者分组，防止数据泄露)...")
    
    # 获取每个样本的患者ID
    patient_ids = full_dataset.get_patient_ids()
    unique_patients = np.array(list(set(patient_ids)))
    print(f"   唯一患者数: {len(unique_patients)}")
    
    # 使用 LabelEncoder 将患者ID转换为数字
    le = LabelEncoder()
    patient_groups = le.fit_transform(patient_ids)
    
    gkf = GroupKFold(n_splits=args.n_folds)
    indices = np.arange(total_samples)
    
    # 保存所有折的划分信息
    all_splits = {
        'n_folds': args.n_folds,
        'seed': args.seed,
        'total_samples': total_samples,
        'total_patients': len(unique_patients),
        'patient_level_split': True,  # 标记使用了患者级别划分
        'folds': []
    }
    
    fold_results = []
    
    for fold, (train_val_idx, test_idx) in enumerate(gkf.split(indices, groups=patient_groups)):
        fold_num = fold + 1
        print(f"\n{'='*60}")
        print(f"Fold {fold_num} / {args.n_folds}")
        print(f"{'='*60}")
        
        # 获取训练+验证集和测试集的患者ID
        train_val_patients = set(patient_ids[i] for i in train_val_idx)
        test_patients = set(patient_ids[i] for i in test_idx)
        
        # 验证没有患者交叉
        overlap = train_val_patients & test_patients
        assert len(overlap) == 0, f"数据泄露! 训练/测试集有 {len(overlap)} 个患者重叠"
        
        # 在 train_val 的患者中按 3:1 划分训练集和验证集
        # 确保按患者级别划分
        train_val_patient_list = list(train_val_patients)
        np.random.seed(args.seed + fold)  # 确保可复现
        np.random.shuffle(train_val_patient_list)
        
        val_patient_count = len(train_val_patient_list) // 4
        val_patients = set(train_val_patient_list[:val_patient_count])
        train_patients = set(train_val_patient_list[val_patient_count:])
        
        # 根据患者ID分配样本
        train_idx = np.array([i for i in train_val_idx if patient_ids[i] in train_patients])
        val_idx = np.array([i for i in train_val_idx if patient_ids[i] in val_patients])
        
        print(f"   - 训练集: {len(train_idx)} 样本 ({len(train_patients)} 患者)")
        print(f"   - 验证集: {len(val_idx)} 样本 ({len(val_patients)} 患者)")
        print(f"   - 测试集: {len(test_idx)} 样本 ({len(test_patients)} 患者)")
        
        # 保存该折的划分
        fold_split = {
            'fold': fold_num,
            'train_idx': train_idx.tolist(),
            'val_idx': val_idx.tolist(),
            'test_idx': test_idx.tolist(),
            'train_patients': len(train_patients),
            'val_patients': len(val_patients),
            'test_patients': len(test_patients)
        }
        all_splits['folds'].append(fold_split)
        
        # 创建 DataLoader
        train_sub = Subset(full_dataset, train_idx)
        val_sub = Subset(full_dataset, val_idx)
        
        train_loader = DataLoader(train_sub, batch_size=args.batch_size, shuffle=True, 
                                  collate_fn=collate_fusion, num_workers=4)
        val_loader = DataLoader(val_sub, batch_size=args.batch_size, shuffle=False, 
                                collate_fn=collate_fusion, num_workers=4)
        
        # 初始化模型
        model = MultiModelFusion(input_dims=input_dims, out_dim=gene_dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        criterion_mse = nn.MSELoss()
        criterion_corr = CorrelationLoss()
        
        # 训练循环
        best_val_p = -1.0
        best_epoch = 0
        
        for epoch in range(args.epochs):
            tr_mse, tr_corrl, tr_p = run_epoch(model, train_loader, criterion_mse, criterion_corr, optimizer, device)
            val_mse, val_corrl, val_p = run_epoch(model, val_loader, criterion_mse, criterion_corr, None, device, is_train=False)
            
            print(f"Epoch {epoch+1:2d} | Train MSE: {tr_mse:.4f}, P: {tr_p:.4f} | Val MSE: {val_mse:.4f}, P: {val_p:.4f}")
            
            if val_p > best_val_p:
                best_val_p = val_p
                best_epoch = epoch + 1
                torch.save(model.state_dict(), os.path.join(args.save_dir, f"best_model_fold{fold_num}.pth"))
        
        fold_results.append({
            'fold': fold_num,
            'best_val_pearson': best_val_p,
            'best_epoch': best_epoch
        })
        print(f"\nFold {fold_num} 最优验证 Pearson: {best_val_p:.4f} (Epoch {best_epoch})")
    
    # 保存所有折的划分信息
    split_path = os.path.join(args.save_dir, 'kfold_splits.json')
    with open(split_path, 'w') as f:
        json.dump(all_splits, f, indent=2)
    print(f"\n划分信息已保存至: {split_path}")
    
    # 汇总结果
    print("\n" + "="*60)
    print("5-Fold 交叉验证训练完成!")
    print("="*60)
    
    val_pearsons = [r['best_val_pearson'] for r in fold_results]
    print(f"\n各折最优验证 Pearson:")
    for r in fold_results:
        print(f"   Fold {r['fold']}: {r['best_val_pearson']:.4f} (Epoch {r['best_epoch']})")
    
    print(f"\n跨折平均验证 Pearson: {np.mean(val_pearsons):.4f} ± {np.std(val_pearsons):.4f}")
    print(f"\n请使用 test_fusion.py 在各折的独立测试集上评估模型。")
    print("="*60)

if __name__ == "__main__":
    main()
