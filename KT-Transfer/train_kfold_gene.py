import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # 可复现更强（可选）

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import numpy as np
import random
import argparse
from tqdm import tqdm
from sklearn.model_selection import GroupKFold

try:
    from sklearn.model_selection import StratifiedGroupKFold
    HAS_SGF = True
except Exception:
    HAS_SGF = False

from dataset import MyDataSet

### [GENE] 新增：读取基因表达矩阵
import pandas as pd


# ==========================================
# 0. Utils
# ==========================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 更严格确定性（如果某些算子报错可先注释掉）
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def seed_worker(worker_id):
    """DataLoader 多 worker 可复现（建议保留）"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def augment_vector(x, drop_prob=0.1, noise_std=0.01):
    # x: [B, Din] slide-level feature
    if drop_prob > 0:
        mask = (torch.rand_like(x) > drop_prob).float()
        x = x * mask
    if noise_std > 0:
        x = x + noise_std * torch.randn_like(x)
    return x


def vicreg_loss(z1, z2, sim_coeff=25.0, std_coeff=25.0, cov_coeff=1.0, eps=1e-4):
    """
    VICReg: invariance + variance + covariance
    z1, z2: [B, D]
    """
    inv = F.mse_loss(z1, z2)

    def std_loss(z):
        std = torch.sqrt(z.var(dim=0) + eps)
        return torch.mean(F.relu(1.0 - std))
    var = std_loss(z1) + std_loss(z2)

    def cov_loss(z):
        z = z - z.mean(dim=0)
        B, D = z.shape
        if B <= 1:
            return torch.tensor(0.0, device=z.device)
        cov = (z.T @ z) / (B - 1)
        off_diag = cov - torch.diag(torch.diag(cov))
        return (off_diag ** 2).sum() / D
    cov = cov_loss(z1) + cov_loss(z2)

    return sim_coeff * inv + std_coeff * var + cov_coeff * cov


def get_beta(epoch, warmup_epochs, ramp_epochs, beta_max):
    # 辅助任务权重：warmup=0，之后线性 ramp 到 beta_max
    if epoch < warmup_epochs:
        return 0.0
    if ramp_epochs <= 0:
        return beta_max
    t = min(1.0, (epoch - warmup_epochs) / float(ramp_epochs))
    return beta_max * t


def get_cv_folds(dataset, n_splits=5, seed=42, label_index_in_samples=1, patient_index_in_samples=2):
    """
    Returns 5 individual folds for manual 3:1:1 logic.
    Each fold is a list of indices.
    """
    groups = []
    y = []
    for s in dataset.samples:
        groups.append(s[patient_index_in_samples])
        try:
            y.append(int(s[label_index_in_samples]))
        except Exception:
            y.append(0)

    groups = np.array(groups)
    y = np.array(y)

    if HAS_SGF:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = splitter.split(np.zeros(len(y)), y, groups)
        print(">>> Using StratifiedGroupKFold (patient-level) for fold generation")
    else:
        splitter = GroupKFold(n_splits=n_splits)
        splits = splitter.split(np.zeros(len(y)), y, groups)
        print(">>> Using GroupKFold (patient-level) for fold generation")

    folds_indices = []
    for _, fold_idx in splits:
        folds_indices.append(fold_idx.tolist())
    
    return folds_indices

# ==========================================
# 1. AttentionPooling (与 models_fusion/fusion_architectures.py 完全一致)
# ==========================================
class AttentionPooling(nn.Module):
    """
    自注意力池化层，将 n 个 d 维向量聚合为一个 d 维向量。
    与 models_fusion/fusion_architectures.py 完全一致，可加载预训练权重。
    """
    def __init__(self, dim=512):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # x: [B, N, D]
        # weights: [B, N, 1]
        weights = self.attention(x)
        weights = F.softmax(weights, dim=1)
        
        # 聚合: [B, 1, N] @ [B, N, D] -> [B, 1, D] -> [B, D]
        fused = torch.bmm(weights.transpose(1, 2), x).squeeze(1)
        return fused, weights


# ==========================================
# 2. MultiModelFusion (教师特征融合模块)
#    结构与 models_fusion/fusion_architectures.py 的 aligners + fusion 部分一致
# ==========================================
class MultiModelFusion(nn.Module):
    """
    多模型融合架构（仅包含特征融合部分）：
    1. 对齐层 (aligners): 映射不同模型的特征到公共 512 维空间。
    2. 注意力池化 (fusion): 融合多个对齐后的特征。
    
    注：不包含回归层，基因预测由 StudentModel.gene_head 完成。
    """
    def __init__(self, input_dims, hidden_dim=512):
        super(MultiModelFusion, self).__init__()
        self.hidden_dim = hidden_dim
        
        # 1. 独立对齐层
        self.aligners = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_d, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ) for in_d in input_dims
        ])
        
        # 2. 注意力融合层
        self.fusion = AttentionPooling(dim=hidden_dim)

    def forward(self, features_list):
        """
        features_list: List of Tensors, each [B, D_i]
        Returns:
            v_fusion: [B, hidden_dim] - 融合后的特征
            attn_weights: [B, N_models, 1] - 注意力权重
        """
        aligned_features = []
        for i, feat in enumerate(features_list):
            aligned_features.append(self.aligners[i](feat))
        
        # 堆叠特征: [B, N_models, hidden_dim]
        stacked = torch.stack(aligned_features, dim=1)
        
        # 注意力融合
        v_fusion, attn_weights = self.fusion(stacked)
        
        return v_fusion, attn_weights

    def load_pretrained(self, pretrained_path, device='cuda'):
        state_dict = torch.load(pretrained_path, map_location=device)

        filtered_state_dict = {
            k: v for k, v in state_dict.items()
            if k.startswith("aligners.") or k.startswith("fusion.")
        }

        missing, unexpected = self.load_state_dict(filtered_state_dict, strict=True)

        print(f">>> Loaded pretrained teacher fusion weights from {pretrained_path}")
        print(f">>> Loaded keys: {list(filtered_state_dict.keys())}")

        if missing:
            print(f">>> Missing keys: {missing}")
        if unexpected:
            print(f">>> Unexpected keys: {unexpected}")


# ==========================================
# 3. StudentModel (KD + Gene regression auxiliary)
#    使用 MultiModelFusion 作为教师融合模块
# ==========================================
class StudentModel(nn.Module):
    def __init__(self, input_dim, teacher_dims, hidden_dim=512, gene_dim=5234):
        super(StudentModel, self).__init__()
        self.hidden_dim = hidden_dim

        # Student encoder
        self.shared_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Student KD head: 映射到与教师融合特征相同的维度
        self.student_kd_head = nn.Linear(hidden_dim, hidden_dim)

        # ### [Teacher Fusion] 教师特征融合模块
        self.teacher_fusion = MultiModelFusion(
            input_dims=teacher_dims,
            hidden_dim=hidden_dim
        )

        # ### [GENE] 基因表达回归头（学生自己的）
        self.gene_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, gene_dim)
        )

    def encode(self, x):
        return self.shared_encoder(x)

    def forward(self, x):
        z = self.shared_encoder(x)              # [B, hidden_dim]
        s_kd = self.student_kd_head(z)          # [B, hidden_dim]
        gene_pred = self.gene_head(z)           # [B, gene_dim]
        return z, s_kd, gene_pred

    def fuse_teachers(self, teacher_batches):
        """
        使用 MultiModelFusion 融合教师特征
        Returns:
            v_fusion: [B, hidden_dim] - 融合后的教师特征
            attn_weights: [B, N_teachers, 1] - 各教师的注意力权重
        """
        v_fusion, attn_weights = self.teacher_fusion(teacher_batches)
        return v_fusion, attn_weights

    def load_teacher_fusion_weights(self, pretrained_path, device='cuda'):
        """
        加载预训练的 MultiModelFusion 权重到 teacher_fusion 模块
        如果预训练模型包含 regressor，会自动忽略
        """
        self.teacher_fusion.load_pretrained(pretrained_path, device)



# ==========================================
# 2. Train / Eval
# ==========================================
def train_one_epoch(
    model, loader, optimizer,
    criterion_gene, device,
    epoch,
    # KD
    kd_weight=1.0,
    # Gene auxiliary schedule
    beta_max=0.2, warmup_epochs=10, ramp_epochs=20,
    # VICReg optional
    use_vicreg=False, gamma=0.0, aug_drop=0.1, aug_noise=0.01,
    teacher_weights=None
):
    model.train()
    total_loss = 0.0
    total_gene = 0.0
    total_kd = 0.0
    total_ssl = 0.0
    total_used = 0

    beta = get_beta(epoch, warmup_epochs, ramp_epochs, beta_max)
    pbar = tqdm(loader, desc=f"Train (beta={beta:.3f})", leave=False)

    for batch in pbar:
        # 新 dataset 输出：s_input, gene_target, patient_ids, teacher_batches
        s_input, gene_target, patient_ids, teacher_batches = batch

        s_input = s_input.to(device)
        gene_target = gene_target.to(device)
        teacher_batches = [t.to(device) for t in teacher_batches]

        # forward
        z, s_kd, gene_pred = model(s_input)

        # ### [GENE] 基因回归 loss（raw）
        loss_gene = criterion_gene(gene_pred, gene_target)

        # KD：使用 AttMIL 融合教师特征，然后计算 cosine loss
        loss_kd = torch.tensor(0.0, device=device)
        num_teachers = len(teacher_batches)
        if num_teachers > 0:
            # 使用注意力池化融合教师特征
            t_consensus, attn_weights = model.fuse_teachers(teacher_batches)
            t_consensus = F.normalize(t_consensus, dim=1)

            s_kd_norm = F.normalize(s_kd, dim=1)
            loss_kd = (1.0 - (s_kd_norm * t_consensus).sum(dim=1)).mean()

        # VICReg optional
        loss_ssl = torch.tensor(0.0, device=device)
        if use_vicreg and gamma > 0:
            x1 = augment_vector(s_input, drop_prob=aug_drop, noise_std=aug_noise)
            x2 = augment_vector(s_input, drop_prob=aug_drop, noise_std=aug_noise)
            z1 = model.encode(x1)
            z2 = model.encode(x2)
            loss_ssl = vicreg_loss(z1, z2)

        # total loss
        loss = kd_weight * loss_kd + beta * loss_gene + gamma * loss_ssl

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_gene += loss_gene.item()
        total_kd += loss_kd.item()
        total_ssl += loss_ssl.item() if (use_vicreg and gamma > 0) else 0.0
        total_used += 1

        # 打印 raw 与加权贡献
        pbar.set_postfix({
            "L": f"{loss.item():.4f}",
            "KD": f"{loss_kd.item():.4f}",
            "Gene": f"{loss_gene.item():.4f}",
            "beta*Gene": f"{(beta*loss_gene).item():.4f}",
            "SSL": f"{loss_ssl.item():.4f}" if (use_vicreg and gamma > 0) else "0.0000",
        })

    n = max(1, total_used)
    return total_loss / n, total_kd / n, total_gene / n, total_ssl / n


def evaluate_gene(model, loader, criterion_gene, device):
    model.eval()
    total_gene = 0.0
    total_used = 0

    with torch.no_grad():
        for batch in loader:
            s_input, gene_target, _, _ = batch
            s_input = s_input.to(device)
            gene_target = gene_target.to(device)

            _, _, gene_pred = model(s_input)
            loss_gene = criterion_gene(gene_pred, gene_target)

            total_gene += loss_gene.item()
            total_used += 1

    n = max(1, total_used)
    return total_gene / n


# ==========================================
# 3. Main: 5-Fold CV (patient-level)
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="5-Fold Patient-level CV | Consensus KD + Gene Aux (ramp) + Optional VICReg")

    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--base_dir', type=str, default="/data/ruiyan/lhj/TRIDENT/BRCA_LUAD_LUSC_STAD_LIHC_BLCA_BALANCED_WSI_FEATURES",help='Base directory containing student and teacher feature folders')
    parser.add_argument('--label_file', type=str, default="/data/ruiyan/lhj/gene_expression_label/combined_all_gene_expression_matrix.csv", help='CSV file containing gene expression labels and patient IDs')
    parser.add_argument('--student_name', type=str, default="titan", help='Name of the student model folder (e.g., titan)')

    parser.add_argument('--teacher_fusion_ckpt', type=str, default='/data/ruiyan/lhj/models_fusion/logs/alldata_fms_checkpoints/best_model_fold1.pth', help='Path to pretrained teacher fusion weights')
 

    # KD / consensus
    parser.add_argument('--consensus_dim', type=int, default=512)
    parser.add_argument('--kd_weight', type=float, default=1.0)

    # Gene auxiliary schedule
    parser.add_argument('--beta_max', type=float, default=0.2, help='Max weight for gene regression loss')
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--ramp_epochs', type=int, default=20)

    # VICReg optional
    parser.add_argument('--use_vicreg', action='store_true')
    parser.add_argument('--gamma', type=float, default=0.0)
    parser.add_argument('--aug_drop', type=float, default=0.1)
    parser.add_argument('--aug_noise', type=float, default=0.01)

    # CV
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--label_index_in_samples', type=int, default=1)
    parser.add_argument('--patient_index_in_samples', type=int, default=2)

    # save
    parser.add_argument('--save_dir', type=str, default="/data/ruiyan/lhj/MFMs-KD/logs/", help="where to save fold checkpoints")

    args = parser.parse_args()
    set_seed(args.seed)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    HIDDEN_DIM = 512

    print(f">>> Device: {DEVICE}")
    print(f">>> epochs={args.epochs}, bs={args.batch_size}, lr={args.lr}, seed={args.seed}")
    print(f">>> KD: consensus_dim={args.consensus_dim}, kd_weight={args.kd_weight}")
    print(f">>> Gene aux: beta_max={args.beta_max}, warmup={args.warmup_epochs}, ramp={args.ramp_epochs}")
    print(f">>> VICReg: use={args.use_vicreg}, gamma={args.gamma}, aug_drop={args.aug_drop}, aug_noise={args.aug_noise}")
    print(f">>> CV: folds={args.folds} (patient-level)")
    print(f">>> Gene CSV: {args.label_file}")
    print(f">>> Save dir: {args.save_dir}")


    base_dir = args.base_dir
    label_file = args.label_file
    student_name = args.student_name
    all_dirs = ['CHIEF', 'Feather', 'madeleine', 'prism', 'titan','GigaPath']
    # all_dirs = ['CHIEF', 'Feather', 'prism', 'titan','GigaPath', 'madeleine']

    student_path = os.path.join(base_dir, student_name)
    # # teacher_names = sorted([d for d in all_dirs if d != student_name])
    # teacher_names = [d for d in all_dirs if d != student_name]
    # teacher_paths_dict = {name: os.path.join(base_dir, name) for name in teacher_names}

    teacher_names = [d for d in all_dirs]
    teacher_paths_dict = {name: os.path.join(base_dir, name) for name in teacher_names}


    print(f">>> Loading Data... Student: {student_name}, Teachers: {teacher_names}")
    full_dataset = MyDataSet(student_path, label_file, teacher_paths_dict, mode='train')
    if len(full_dataset) == 0:
        print("Empty dataset.")
        return

    # detect dims
    print(">>> Detecting Dimensions...")
    sample = full_dataset[0]
    input_dim = sample[0].shape[-1]

    if len(sample) > 3 and isinstance(sample[3], (list, tuple)):
        teacher_dims = [t.shape[-1] for t in sample[3]]
    else:
        teacher_dims = [sample[3 + i].shape[-1] for i in range(len(teacher_names))]

    print(f"  Input Dim: {input_dim}")
    print(f"  Teacher Dims: {teacher_dims}")

    # ### [GENE] get gene dimension from dataset
    gene_dim = full_dataset.gene_dim
    print(f"  Gene Dimension: {gene_dim}")

    # get individual folds
    folds_indices = get_cv_folds(
        full_dataset,
        n_splits=args.folds,
        seed=args.seed,
        label_index_in_samples=args.label_index_in_samples,
        patient_index_in_samples=args.patient_index_in_samples
    )

    save_dir =  os.path.join(args.save_dir, student_name)
    splits_dir = os.path.join(save_dir, "splits")
    os.makedirs(splits_dir, exist_ok=True)
    print(f">>> Splits will be saved to: {splits_dir}")

    # ### [GENE] regression loss：SmoothL1 更稳（也可换 MSELoss）
    criterion_gene = nn.SmoothL1Loss(beta=1.0)

    print("\n>>> Start CV Training...")
    fold_best_gene_losses = []
    fold_test_gene_losses = []

    # Map dataset index to patient_id for split saving
    index2patient = {i: full_dataset.samples[i][2] for i in range(len(full_dataset.samples))}

    for fold_id in range(args.folds):
        print(f"\n========== Fold {fold_id}/{args.folds-1} ==========")

        # 3:1:1 split (Rotation: Test=k, Val=k+1, Train=others)
        test_indices = folds_indices[fold_id]
        val_indices = folds_indices[(fold_id + 1) % args.folds]
        
        train_indices = []
        for i in range(args.folds):
            if i != fold_id and i != (fold_id + 1) % args.folds:
                train_indices.extend(folds_indices[i])

        # Save splits (PATIENT ID format)
        train_p = [index2patient[i] for i in train_indices]
        val_p = [index2patient[i] for i in val_indices]
        test_p = [index2patient[i] for i in test_indices]
        
        pd.DataFrame({'PATIENT': train_p}).to_csv(os.path.join(splits_dir, f"fold{fold_id}_train.csv"))
        pd.DataFrame({'PATIENT': val_p}).to_csv(os.path.join(splits_dir, f"fold{fold_id}_val.csv"))
        pd.DataFrame({'PATIENT': test_p}).to_csv(os.path.join(splits_dir, f"fold{fold_id}_test.csv"))
        print(f"  Splits (train:val:test = 3:1:1) saved to {splits_dir}")

        train_set = Subset(full_dataset, train_indices)
        val_set = Subset(full_dataset, val_indices)
        test_set = Subset(full_dataset, test_indices)

        g = torch.Generator()
        g.manual_seed(args.seed + fold_id)

        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=MyDataSet.collate_fn,
            num_workers=4,
            worker_init_fn=seed_worker,
            generator=g
        )
        val_loader = DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=MyDataSet.collate_fn,
            num_workers=4,
            worker_init_fn=seed_worker,
            generator=g
        )
        test_loader = DataLoader(
            test_set,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=MyDataSet.collate_fn,
            num_workers=4,
            worker_init_fn=seed_worker,
            generator=g
        )

        model = StudentModel(
            input_dim=input_dim,
            teacher_dims=teacher_dims,
            hidden_dim=HIDDEN_DIM,
            gene_dim=gene_dim
        ).to(DEVICE)



        if args.teacher_fusion_ckpt:
            model.load_teacher_fusion_weights(args.teacher_fusion_ckpt, device=DEVICE)

        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        best_val_gene = float("inf")
        best_path = os.path.join(args.save_dir, f"best_student_fold{fold_id}.pth")

        for epoch in range(args.epochs):
            tr_loss, tr_kd, tr_gene, tr_ssl = train_one_epoch(
                model, train_loader, optimizer,
                criterion_gene, DEVICE,
                epoch=epoch,
                kd_weight=args.kd_weight,
                beta_max=args.beta_max,
                warmup_epochs=args.warmup_epochs,
                ramp_epochs=args.ramp_epochs,
                use_vicreg=args.use_vicreg,
                gamma=args.gamma,
                aug_drop=args.aug_drop,
                aug_noise=args.aug_noise,
                teacher_weights=None
            )

            val_gene = evaluate_gene(model, val_loader, criterion_gene, DEVICE)

            print(f"Fold{fold_id} Ep[{epoch+1}/{args.epochs}] "
                  f"Tr: L={tr_loss:.3f} KD={tr_kd:.3f} Gene={tr_gene:.3f} SSL={tr_ssl:.3f} | "
                  f"Val GeneLoss={val_gene:.4f}")

            # 以 gene loss 作为保存 best 的标准（你也可以改成 kd+gene 的组合）
            if val_gene < best_val_gene:
                best_val_gene = val_gene
                torch.save(model.state_dict(), best_path)

        # fold best evaluation
        model.load_state_dict(torch.load(best_path, map_location=DEVICE))
        fold_v_loss = evaluate_gene(model, val_loader, criterion_gene, DEVICE)
        fold_t_loss = evaluate_gene(model, test_loader, criterion_gene, DEVICE)
        
        fold_best_gene_losses.append(fold_v_loss)
        fold_test_gene_losses.append(fold_t_loss)
        
        print(f">>> Fold {fold_id} Best Val GeneLoss: {fold_v_loss:.4f} | Test GeneLoss: {fold_t_loss:.4f}")

    mean_v = float(np.mean(fold_best_gene_losses))
    std_v = float(np.std(fold_best_gene_losses))
    mean_t = float(np.mean(fold_test_gene_losses))
    std_t = float(np.std(fold_test_gene_losses))

    print("\n========== CV Summary ==========")
    for i, (vl, tl) in enumerate(zip(fold_best_gene_losses, fold_test_gene_losses)):
        print(f"Fold {i}: Val={vl:.4f}, Test={tl:.4f}")
    print(f"Mean Best Val GeneLoss  = {mean_v:.4f} ± {std_v:.4f}")
    print(f"Mean Final Test GeneLoss = {mean_t:.4f} ± {std_t:.4f}")


if __name__ == "__main__":
    main()

