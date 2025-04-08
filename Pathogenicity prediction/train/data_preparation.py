import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold, train_test_split


# Set random seed
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(20)

data_file = '/mnt/data0/users/baiy/CNV/data/h5_file/train_data_0.956_best_model/673_data/Clinvar_30_brain_dieases_cnv_HG38_noncoding_variants_146_nc527_673_train_log2_scores_LR.h5'
labels_file = '/mnt/data0/users/baiy/CNV/data/h5_file/train_data_0.956_best_model/673_data/Clinvar_30_brain_dieases_cnv_HG38_noncoding_variants_146_nc527_673_labels.h5'
csv_file = '/mnt/data0/users/baiy/CNV/data/Brain_CNV_data/Clinvar_30_brain_dieases_cnv_HG38_noncoding_variants_146_nc527_673.csv'
df = pd.read_csv(csv_file)

# Load data and labels from .h5 files
with h5py.File(data_file, 'r') as f:
    data = np.asarray(f['log2_scores'], dtype=np.float32)

with h5py.File(labels_file, 'r') as f:
    labels = np.asarray(f['Labels'], dtype=np.float32)

original_indices = np.arange(len(data))

kf = KFold(n_splits=5, shuffle=True, random_state=20)

for fold, (train_val_indices, test_indices) in enumerate(kf.split(data, labels)):
    print(f'Preparing Fold {fold + 1}')

    data_train_val, data_test = data[train_val_indices], data[test_indices]
    label_train_val, label_test = labels[train_val_indices], labels[test_indices]

    test_indices_in_original = original_indices[test_indices]
    train_val_indices_in_original = original_indices[train_val_indices]

    train_data, val_data, train_labels, val_labels, train_indices, val_indices= train_test_split(
        data_train_val, label_train_val, train_val_indices_in_original, test_size=0.25, stratify=label_train_val, random_state=20
    )

    train_data_csv = df.iloc[train_indices]
    val_data_csv = df.iloc[val_indices]
    test_data_csv = df.iloc[test_indices_in_original]

    # Save training data as CSV file
    train_data_csv.to_csv(f'/mnt/data0/users/baiy/CNV/data/Brain_CNV_data/five_fold_dataset_0.956_best_model/673_LR_five_fold_data/train_data/fold_{fold + 1}_train_data.csv', index=False)
    val_data_csv.to_csv(f'/mnt/data0/users/baiy/CNV/data/Brain_CNV_data/five_fold_dataset_0.956_best_model/673_LR_five_fold_data/val_data/fold_{fold + 1}_val_data.csv', index=False)
    test_data_csv.to_csv(f'/mnt/data0/users/baiy/CNV/data/Brain_CNV_data/five_fold_dataset_0.956_best_model/673_LR_five_fold_data/test_data/fold_{fold + 1}_test_data.csv', index=False)

    # Save data as.H5 file
    with h5py.File(f"/mnt/data0/users/baiy/CNV/data/Brain_CNV_data/five_fold_dataset_0.956_best_model/673_LR_five_fold_data/train_data/fold_{fold + 1}_train_data.h5", 'w') as f_train:
        f_train.create_dataset('data', data=train_data)
        f_train.create_dataset('labels', data=train_labels.astype(int))

    with h5py.File(f"/mnt/data0/users/baiy/CNV/data/Brain_CNV_data/five_fold_dataset_0.956_best_model/673_LR_five_fold_data/val_data/fold_{fold + 1}_val_data.h5", 'w') as f_val:
        f_val.create_dataset('data', data=val_data)
        f_val.create_dataset('labels', data=val_labels.astype(int))

    with h5py.File(f"/mnt/data0/users/baiy/CNV/data/Brain_CNV_data/five_fold_dataset_0.956_best_model/673_LR_five_fold_data/test_data/fold_{fold + 1}_test_data.h5", 'w') as f_test:
        f_test.create_dataset('data', data=data[test_indices_in_original])
        f_test.create_dataset('labels', data=labels[test_indices_in_original].astype(int))

print("Data preparation complete.")
