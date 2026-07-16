import torch
from torch.utils.data import Dataset, DataLoader
import os
import h5py
import numpy as np
import random
import shutil
import pandas as pd


def tcga_case_id(x: str) -> str:
    """将 TCGA barcode 转换为 12 位 case_id"""
    if x is None:
        return None
    x = str(x).strip().replace(".", "-")   # '.' -> '-'
    if x.upper().startswith("TCGA-") and len(x) >= 12:
        return x[:12]
    return x


class MyDataSet(Dataset):
    """
    Dataset for Multi-Teacher Knowledge Distillation with Gene Expression Labels.
    Returns: Student_Feature, Gene_Expression_Vector, Patient_ID, [Teacher_Features...]
    
    label_path: path to gene expression matrix CSV (genes x samples)
    """

    def __init__(self, student_dir, label_path, 
                 teacher_paths_dict, 
                 mode='train', transform=None):
        
        self.samples = []
        self.labels = {}  # case_id -> gene expression vector
        self.teacher_emds = {name: {} for name in teacher_paths_dict.keys()} 
        
        self.student_dir = student_dir
        self.label_path = label_path
        self.teacher_paths_dict = teacher_paths_dict
        self.mode = mode
        self.transform = transform
        
        self.gene_names = None  # 存储基因名列表
        self.gene_dim = None    # 基因维度
        
        print(f"[{mode}] Initializing Dataset...")
        self._load_gene_expression_labels()
        
        for teacher_name, teacher_path in self.teacher_paths_dict.items():
            self._load_teacher_emds_generic(teacher_name, teacher_path)
            
        self._pre_process()

        # 记录特征维度以备 fallback 使用
        self.student_dim = 768  # 默认 TITAN 维度
        if len(self.samples) > 0:
            s_path = self.samples[0][0]
            try:
                with h5py.File(s_path, 'r') as f:
                    key = 'features' if 'features' in f else list(f.keys())[0]
                    self.student_dim = f[key].shape[-1]
            except Exception: pass

    def _load_gene_expression_labels(self):
        """
        加载基因表达矩阵作为标签
        期望格式：行=基因，列=样本（TCGA barcode），值=normalized expression
        返回：case_id -> np.ndarray (gene_dim,)
        """
        if not os.path.exists(self.label_path):
            raise FileNotFoundError(f"Label file not found: {self.label_path}")
        
        print(f"[{self.mode}] Loading gene expression matrix from: {self.label_path}")
        df = pd.read_csv(self.label_path, index_col=0)
        
        self.gene_names = df.index.tolist()  # [G]
        self.gene_dim = len(self.gene_names)
        col_ids = df.columns.tolist()        # samples
        mat = df.values.T.astype(np.float32) # [N, G] (转置成样本在前)
        
        # 构建 case_id -> expression 的映射
        # 如果同一个 case 对应多个 aliquot，做平均
        case2list = {}
        for i, sid in enumerate(col_ids):
            cid = tcga_case_id(sid)
            case2list.setdefault(cid, []).append(mat[i])
        
        self.labels = {cid: np.mean(np.stack(v, axis=0), axis=0) 
                       for cid, v in case2list.items()}
        
        print(f"[{self.mode}] Loaded gene matrix: genes={self.gene_dim}, "
              f"samples={len(col_ids)}, unique_cases={len(self.labels)}")

    def _parse_slide_id(self, filename):
        return filename.split('.')[0]

    def _load_teacher_emds_generic(self, teacher_name, emds_path):
        if not os.path.exists(emds_path): return

        files = [f for f in os.listdir(emds_path) if f.endswith('.h5')]
        for fname in files:
            fpath = os.path.join(emds_path, fname)
            try:
                wsi_id = self._parse_slide_id(fname)
                with h5py.File(fpath, "r") as f:
                    keys = list(f.keys())
                    key = 'features' if 'features' in keys else keys[1]
                    data = np.array(f[key])
                    self.teacher_emds[teacher_name][wsi_id] = data
            except Exception: pass

    def _pre_process(self):
        if not os.path.exists(self.student_dir): return

        student_files = [f for f in os.listdir(self.student_dir) if f.endswith('.h5')]
        skipped_no_label = 0
        skipped_no_teacher = 0
        
        for s_file in student_files:
            # 1. 解析 ID
            wsi_id = self._parse_slide_id(s_file)  # TCGA-VQ-A94U-01Z-00-DX1
            patient_id = tcga_case_id(wsi_id)      # TCGA-VQ-A94U
            
            # 2. 匹配基因表达 Label
            if patient_id not in self.labels:
                skipped_no_label += 1
                continue
            
            # 3. 匹配 Teacher
            missing_teacher = False
            teacher_feats_list = []
            
            for t_name in self.teacher_paths_dict.keys():
                if wsi_id not in self.teacher_emds[t_name]:
                    missing_teacher = True
                    break
                teacher_feats_list.append(self.teacher_emds[t_name][wsi_id])
            
            if missing_teacher:
                skipped_no_teacher += 1
                continue

            # --- 加入样本 ---
            # label 现在是基因表达向量
            gene_expr = self.labels[patient_id]  # shape: (gene_dim,)
            s_path = os.path.join(self.student_dir, s_file)
            self.samples.append((s_path, gene_expr, patient_id, *teacher_feats_list))
            
        if self.mode == 'train':
            random.shuffle(self.samples)
        
        print(f"[{self.mode}] Valid samples: {len(self.samples)}, "
              f"skipped (no label): {skipped_no_label}, "
              f"skipped (no teacher): {skipped_no_teacher}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        
        # 解包
        s_path = sample[0]
        gene_expr = sample[1]      # np.ndarray (gene_dim,)
        patient_id = sample[2]     # str
        
        # 读取 Student
        try:
            with h5py.File(s_path, 'r') as f:
                key = 'features' if 'features' in f else list(f.keys())[0]
                student_feat = np.array(f[key])
        except Exception:
            student_feat = np.zeros(self.student_dim)
        
        # 统一形状为 [D]
        student_feat = student_feat.squeeze() 
        
        if self.transform:
            student_feat = self.transform(student_feat)
            
        student_feat = torch.Tensor(student_feat)
        
        # 统一 Teacher 特征形状
        teacher_feats = [t.squeeze() for t in sample[3:]]
        
        # 返回：student_feat, gene_expr, patient_id, *teacher_features
        return student_feat, gene_expr, patient_id, *teacher_feats

    @staticmethod
    def collate_fn(batch):
        """
        Collate function for DataLoader.
        Returns: student_feats, gene_labels, patient_ids, teacher_batches
        """
        # item[0]: student, item[1]: gene_expr, item[2]: patient_id, item[3:]: teachers
        
        student_feats = torch.stack([item[0] for item in batch], dim=0)
        
        # gene expression labels: stack into tensor
        gene_labels = torch.from_numpy(np.stack([item[1] for item in batch], axis=0))
        
        # patient_ids: 字符串列表
        patient_ids = [item[2] for item in batch]
        
        # 处理 Teacher (索引从 3 开始)
        num_teachers = len(batch[0]) - 3
        teacher_batches = []
        
        for t_idx in range(num_teachers):
            t_data = [item[3 + t_idx] for item in batch]
            # 确保 t_data 中所有元素形状一致
            t_stack = torch.as_tensor(np.stack(t_data))
            teacher_batches.append(t_stack)
            
        return student_feats, gene_labels, patient_ids, teacher_batches


# ==========================================
#  Main Execution (Test)
# ==========================================
if __name__ == "__main__": 
    # --- 配置路径 ---
    base_dir = "/data/ruiyan/lhj/TRIDENT/BALANCED_WSI_FEATURES"
    label_file = "/data/ruiyan/lhj/gene_expression_label/combined_subset_gene_expression_matrix.csv"
    
    student_name = "titan"
    all_dirs = ['CHIEF', 'Feather', 'madeleine', 'prism', 'titan']
    
    student_path = os.path.join(base_dir, student_name)
    teacher_names = [d for d in all_dirs if d != student_name]
    teacher_paths_dict = {name: os.path.join(base_dir, name) for name in teacher_names}
    print("teacher_paths_dict:", teacher_paths_dict)
    
    # --- 运行 ---
    dataset = MyDataSet(
        student_dir=student_path,
        label_path=label_file,
        teacher_paths_dict=teacher_paths_dict,
        mode='train'
    )

    if len(dataset) > 0:
        print(f"\nGene dimension: {dataset.gene_dim}")
        print(f"First 5 gene names: {dataset.gene_names[:5]}")
        
        loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=MyDataSet.collate_fn)
        for batch in loader:
            s_feat, gene_labels, p_ids, teachers = batch
            
            print("\n--- Batch Info ---")
            print(f"Student Shape: {s_feat.shape}")
            print(f"Gene Labels Shape: {gene_labels.shape}")  # [B, gene_dim]
            print(f"Gene Labels dtype: {gene_labels.dtype}")
            print(f"Patient IDs: {p_ids}")
            print(f"Teacher Count: {len(teachers)}")
            break
    else:
        print("No valid samples found.")
