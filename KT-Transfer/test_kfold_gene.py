import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_squared_error

from dataset import MyDataSet
from train_kfold_gene import StudentModel

def evaluate_test(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Testing", leave=False):
            s_input, gene_target, _, _ = batch
            s_input = s_input.to(device)
            gene_target = gene_target.to(device)

            _, _, gene_pred = model(s_input)
            
            all_preds.append(gene_pred.cpu().numpy())
            all_targets.append(gene_target.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    return all_preds, all_targets

def main():
    parser = argparse.ArgumentParser(description="Test K-Fold Gene Expression Model (3:1:1 Split)")
    parser.add_argument('--save_dir', type=str, default="/data/ruiyan/lhj/MFMs-KD/checkpoints_gene", help="Directory where checkpoints and splits are saved")
    # parser.add_argument('--gene_csv', type=str, required=True, help="Path to gene expression matrix csv")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu")


    parser.add_argument('--base_dir', type=str, default="/data/ruiyan/lhj/TRIDENT/BRCA_LUAD_LUSC_STAD_LIHC_BLCA_BALANCED_WSI_FEATURES",help='Base directory containing student and teacher feature folders')
    parser.add_argument('--label_file', type=str, default="/data/ruiyan/lhj/gene_expression_label/combined_all_gene_expression_matrix.csv", help='CSV file containing gene expression labels and patient IDs')
    parser.add_argument('--student_name', type=str, default="titan", help='Name of the student model folder (e.g., titan)')

    args = parser.parse_args()
    DEVICE = torch.device(args.device)
    HIDDEN_DIM = 512

    # ====== Paths & Settings (Align with train_kfold_gene.py) ======
    # base_dir = "/data/ruiyan/lhj/TRIDENT/BALANCED_WSI_FEATURES"
    # student_name = "titan"
    base_dir = args.base_dir
    label_file = args.label_file
    student_name = args.student_name
    
    all_dirs = ['CHIEF', 'Feather', 'madeleine', 'prism', 'titan','GigaPath']

    student_path = os.path.join(base_dir, student_name)
    teacher_names = sorted([d for d in all_dirs if d != student_name])
    teacher_paths_dict = {name: os.path.join(base_dir, name) for name in teacher_names}

    print(f">>> Loading Dataset...")
    full_dataset = MyDataSet(student_path, args.label_file, teacher_paths_dict, mode='test')
    
    # detect dims
    sample = full_dataset[0]
    input_dim = sample[0].shape[-1]
    
    # Handle the structure of sample[3]
    if len(sample) > 3 and isinstance(sample[3], (list, tuple)):
        teacher_dims = [t.shape[-1] for t in sample[3]]
    else:
        teacher_dims = [sample[3+i].shape[-1] for i in range(len(teacher_names))]
        
    gene_dim = full_dataset.gene_dim

    splits_dir = os.path.join(args.save_dir, "splits")
    if not os.path.exists(splits_dir):
        raise FileNotFoundError(f"Splits directory not found in {args.save_dir}")

    # Find test splits (looking for foldX_test.csv)
    test_files = sorted([f for f in os.listdir(splits_dir) if f.endswith("_test.csv")])
    num_folds = len(test_files)
    print(f">>> Found {num_folds} test splits.")

    overall_metrics = []

    for f_name in test_files:
        # Extract fold_id (e.g. fold0, fold1)
        fold_label = f_name.split('_')[0]
        print(f"\n========== Testing {fold_label} ==========")
        
        # 1. Load split by PATIENT IDs
        df = pd.read_csv(os.path.join(splits_dir, f_name))
        if 'PATIENT' not in df.columns:
            print(f"  Warning: PATIENT column not found in {f_name}, skipping.")
            continue
            
        test_patients = set(df['PATIENT'].astype(str).tolist())
        
        # Map back to dataset indices
        test_indices = [i for i, s in enumerate(full_dataset.samples) if s[2] in test_patients]
        
        if not test_indices:
            print(f"  Warning: No samples found for patients in {f_name}, skipping.")
            continue
            
        test_set = Subset(full_dataset, test_indices)
        
        loader = DataLoader(
            test_set, 
            batch_size=args.batch_size, 
            shuffle=False, 
            collate_fn=MyDataSet.collate_fn,
            num_workers=4
        )

        # 2. Load model
        model = StudentModel(
            input_dim=input_dim,
            teacher_dims=teacher_dims,
            hidden_dim=HIDDEN_DIM,
            gene_dim=gene_dim
        ).to(DEVICE)
        
        # Checkpoint naming: best_student_foldX.pth (0-indexed)
        fold_id_str = fold_label.replace("fold", "")
        best_path = os.path.join(args.save_dir, f"best_student_fold{fold_id_str}.pth")
        
        if not os.path.exists(best_path):
            print(f"  Warning: Checkpoint not found at {best_path}, skipping.")
            continue
            
        model.load_state_dict(torch.load(best_path, map_location=DEVICE))
        
        # 3. Inference
        preds, targets = evaluate_test(model, loader, DEVICE)
        
        # 4. Metrics
        mse = mean_squared_error(targets, preds)
        r2 = r2_score(targets, preds)
        
        pearsons = []
        for i in range(gene_dim):
            # Only calculate if there is variance
            if np.std(targets[:, i]) > 0 and np.std(preds[:, i]) > 0:
                p_val, _ = pearsonr(targets[:, i], preds[:, i])
                if not np.isnan(p_val):
                    pearsons.append(p_val)
        
        avg_pearson = np.mean(pearsons) if pearsons else 0.0
        
        print(f"  MSE: {mse:.4f}")
        print(f"  R2:  {r2:.4f}")
        print(f"  Avg Pearson: {avg_pearson:.4f}")
        
        overall_metrics.append({
            'fold': fold_label,
            'mse': mse,
            'r2': r2,
            'avg_pearson': avg_pearson
        })

    # Summary
    if overall_metrics:
        df_results = pd.DataFrame(overall_metrics)
        print("\n========== Test Summary ==========")
        print(df_results)
        print("\nMean Metrics:")
        print(df_results.mean(numeric_only=True))
        
        # Save results
        results_path = os.path.join(args.save_dir, "test_results.csv")
        df_results.to_csv(results_path, index=False)
        print(f"\nFinal results saved to {results_path}")

if __name__ == "__main__":
    main()
