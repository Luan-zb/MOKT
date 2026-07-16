import os
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class FusionDataset(Dataset):
    """
    多模型融合数据集：
    读取多个模型的 .h5 特征，并匹配基因表达真值。
    """
    def __init__(self, base_models_dir, gene_expr_path, model_names=['CHIEF', 'Feather', 'madeleine', 'prism','titan'], transform=None):
        self.base_models_dir = base_models_dir
        self.model_names = model_names
        self.transform = transform
        
        # 1. 加载基因表达数据 (行=基因, 列=样本)
        print(f"Loading gene expression from {gene_expr_path}...")
        self.gene_df = pd.read_csv(gene_expr_path, index_col=0)
        self.gene_dim = self.gene_df.shape[0]
        
        # 建立 Case ID -> Expression 映射
        self.case2expr = {}
        for col in self.gene_df.columns:
            case_id = self._format_tcga_id(col)
            # 如果有多个 aliquot，取平均（或覆盖，此处简单处理）
            expr = self.gene_df[col].values.astype(np.float32)
            self.case2expr[case_id] = expr
        
        print(f"Loaded {len(self.case2expr)} gene expression cases.")
        
        # 2. 扫描共享样本
        # 以第一个模型目录为基准
        ref_model_path = os.path.join(base_models_dir, model_names[0])
        all_h5_files = sorted([f for f in os.listdir(ref_model_path) if f.endswith('.h5')])
        
        self.samples = []
        skipped_count = 0
        for f in all_h5_files:
            case_id = self._format_tcga_id(f)
            if case_id in self.case2expr:
                # 验证所有模型的 h5 文件是否存在且可读
                valid = True
                for model in model_names:
                    h5_path = os.path.join(base_models_dir, model, f)
                    if not os.path.exists(h5_path):
                        valid = False
                        break
                    try:
                        with h5py.File(h5_path, 'r') as hf:
                            key = 'features' if 'features' in hf else list(hf.keys())[0]
                            _ = hf[key].shape  # 尝试读取确保文件完整
                    except Exception as e:
                        print(f"Skipping corrupted file: {model}/{f} - {e}")
                        valid = False
                        break
                
                if valid:
                    # 提取患者ID (TCGA-XX-XXXX) 用于分组划分
                    patient_id = self._get_patient_id(f)
                    self.samples.append({
                        'filename': f,
                        'case_id': case_id,
                        'patient_id': patient_id,
                        'label': self.case2expr[case_id]
                    })
                else:
                    skipped_count += 1
        
        # 统计患者信息
        unique_patients = set(s['patient_id'] for s in self.samples)
        print(f"Final dataset size: {len(self.samples)} (with matching gene labels)")
        print(f"Unique patients: {len(unique_patients)}")
        if skipped_count > 0:
            print(f"Skipped {skipped_count} samples due to missing/corrupted h5 files")

    def _get_patient_id(self, name):
        """从文件名提取患者ID (TCGA-XX-XXXX)"""
        name = name.replace('.', '-')
        parts = name.split('-')
        if len(parts) >= 3:
            return "-".join(parts[:3])
        return name
    
    def get_patient_ids(self):
        """返回所有样本对应的患者ID列表，用于 GroupKFold"""
        return [s['patient_id'] for s in self.samples]

    def _format_tcga_id(self, name):
        """将 TCGA.XX.XXXX... 或 TCGA-XX-XXXX... 统一转化为 TCGA-XX-XXXX"""
        name = name.replace('.', '-')
        parts = name.split('-')
        if len(parts) >= 3:
            return "-".join(parts[:3])
        return name

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        filename = item['filename']
        label = item['label']
        
        # 加载所有模型的特征
        features_list = []
        for model in self.model_names:
            h5_path = os.path.join(self.base_models_dir, model, filename)
            with h5py.File(h5_path, 'r') as f:
                # 假设数据集名称为 'features'
                key = 'features' if 'features' in f else list(f.keys())[0]
                feat = np.array(f[key]).astype(np.float32)
                # 确保是向量 (某些模型可能输出 [1, D] 或 [D])
                feat = feat.flatten()
                features_list.append(torch.from_numpy(feat))
        
        label_tensor = torch.from_numpy(label)
        
        return features_list, label_tensor, item['case_id']

def collate_fusion(batch):
    # batch: List of (features_list, label, case_id)
    num_models = len(batch[0][0])
    
    # 将每个模型的特征分别打包
    model_batches = []
    for m_idx in range(num_models):
        m_feats = torch.stack([item[0][m_idx] for item in batch])
        model_batches.append(m_feats)
        
    labels = torch.stack([item[1] for item in batch])
    case_ids = [item[2] for item in batch]
    
    return model_batches, labels, case_ids

if __name__ == "__main__":
    # 测试代码
    data_dir = "/data/ruiyan/lhj/TRIDENT/BALANCED_WSI_FEATURES"
    gene_path = "/data/ruiyan/lhj/gene_expression_label/combined_subset_gene_expression_matrix.csv"
    
    dataset = FusionDataset(data_dir, gene_path)
    if len(dataset) > 0:
        loader = torch.utils.data.DataLoader(dataset, batch_size=4, collate_fn=collate_fusion)
        for m_batch, labels, ids in loader:
            print(f"Batch Model 1 Shape: {m_batch[0].shape}")
            print(f"Labels Shape: {labels.shape}")
            print(f"Case IDs: {ids}")
            break
