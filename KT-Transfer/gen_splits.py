import os
import torch
import numpy as np
import pandas as pd
import sys

# Ensure current directory is in path
sys.path.append(os.getcwd())

from dataset import MyDataSet
from train_kfold_gene import get_cv_splits_by_patient

base_dir = '/data/ruiyan/lhj/TRIDENT/BALANCED_WSI_FEATURES'
label_file = '/data/ruiyan/lhj/gene_expression_label/combined_subset_gene_expression_matrix.csv'
student_name = 'titan'
all_dirs = ['CHIEF', 'Feather', 'madeleine', 'prism', 'titan']
student_path = os.path.join(base_dir, student_name)
teacher_names = sorted([d for d in all_dirs if d != student_name])
teacher_paths_dict = {name: os.path.join(base_dir, name) for name in teacher_names}

print("Initializing dataset...")
full_dataset = MyDataSet(student_path, label_file, teacher_paths_dict, mode='train')
print("Calculating splits...")
folds = get_cv_splits_by_patient(full_dataset, n_splits=5, seed=42)

splits_dir = './test_checkpoints/splits'
os.makedirs(splits_dir, exist_ok=True)

for fold_id, (train_indices, val_indices) in enumerate(folds, start=1):
    pd.DataFrame({'index': train_indices}).to_csv(os.path.join(splits_dir, f'fold{fold_id}_train.csv'), index=False)
    pd.DataFrame({'index': val_indices}).to_csv(os.path.join(splits_dir, f'fold{fold_id}_val.csv'), index=False)
    print(f'Fold {fold_id} indices saved to {splits_dir}')
