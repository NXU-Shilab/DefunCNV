import random
import joblib
import torch
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, precision_recall_curve, auc
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# Set random seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(20)

# Load data function
def load_data(fold):

    with h5py.File(
            f"/mnt/data0/users/baiy/CNV/data/Brain_CNV_data/five_fold_dataset_0.956_best_model/673_LR_five_fold_data/val_data/fold_{fold + 1}_val_data.h5",
            'r') as f_val:
        val_data = np.array(f_val['data'])
        val_labels = np.array(f_val['labels'])

    with h5py.File(
            f"/mnt/data0/users/baiy/CNV/data/Brain_CNV_data/five_fold_dataset_0.956_best_model/673_LR_five_fold_data/test_data/fold_{fold + 1}_test_data.h5",
            'r') as f_test:
        test_data = np.array(f_test['data'])
        test_labels = np.array(f_test['labels'])

    return val_data, val_labels, test_data, test_labels



def evaluate_model(model, data, labels):

    predictions = model.predict_proba(data)[:, 1]
    true_labels = labels

    # ROC curves were calculated to obtain TPR and FPR
    fpr, tpr, thresholds = roc_curve(true_labels, predictions)
    finite_indices = np.isfinite(thresholds)
    finite_thresholds = thresholds[finite_indices]
    finite_youden_index = (tpr - fpr)[finite_indices]
    best_index = np.argmax(finite_youden_index)
    best_threshold = finite_thresholds[best_index]
    best_threshold = round(best_threshold, 3)
    # Save optimal threshold
    np.save(f'/mnt/data0/users/baiy/CNV/data/Brain_CNV_data/five_fold_dataset_0.956_best_model/673_LR_five_fold_data/five_fold_best_param/best_threshold_fold_{fold + 1}.npy', best_threshold)
    AUC = roc_auc_score(true_labels, predictions)
    precision, recall, _ = precision_recall_curve(true_labels, predictions)
    AUPRC = auc(recall, precision)
    ACC = accuracy_score(true_labels, [1 if o > best_threshold else 0 for o in predictions])

    return AUC, AUPRC, ACC

# Load best hyperparameter file
best_params_df = pd.read_csv(
    '/mnt/data0/users/baiy/CNV/data/Brain_CNV_data/five_fold_dataset_0.956_best_model/673_LR_five_fold_data/five_fold_best_param/fold_results_rf.csv')

val_auc_folds = []
val_auprc_folds = []
val_acc_folds = []
test_auc_folds = []
test_auprc_folds = []
test_acc_folds = []

print("======Load models and evaluate on validation and test sets======")

# Cycle every fold
for fold in range(5):
    print(f'========== Test Fold {fold + 1} ==========')

    val_data, val_labels, test_data, test_labels = load_data(fold)

    best_params = best_params_df.iloc[fold]

    best_model_path = f'/mnt/data0/users/baiy/CNV/data/Brain_CNV_data/five_fold_dataset_0.956_best_model/673_LR_five_fold_data/five_fold_best_param/rf_best_models/fold_{fold + 1}_best_model_rf.pkl'

    model = joblib.load(best_model_path)

    val_auc, val_auprc, val_acc = evaluate_model(model, val_data, val_labels)
    val_auc_folds.append(val_auc)
    val_auprc_folds.append(val_auprc)
    val_acc_folds.append(val_acc)

    test_auc, test_auprc, test_acc = evaluate_model(model, test_data, test_labels)
    test_auc_folds.append(test_auc)
    test_auprc_folds.append(test_auprc)
    test_acc_folds.append(test_acc)

mean_val_auc = np.mean(val_auc_folds)
mean_val_auprc = np.mean(val_auprc_folds)
mean_test_auc = np.mean(test_auc_folds)
mean_test_auprc = np.mean(test_auprc_folds)
mean_test_acc = np.mean(test_acc_folds)

print(f'5-Fold Validation AUC Mean: {mean_val_auc:.4f}')
print(f'5-Fold Validation AUPRC Mean: {mean_val_auprc:.4f}')
print(f'5-Fold Test AUC Mean: {mean_test_auc:.4f}')
print(f'5-Fold Test AUPRC Mean: {mean_test_auprc:.4f}')
print(f'5-Fold Test Accuracy Mean: {mean_test_acc:.4f}')

# data2638_color
# val_auc_color = '#1f77b4'
# test_auc_color = '#ff7f0e'
# test_acc_color = '#d62728'

# data2638LR_color
# val_auc_color = '#9467bd'
# test_auc_color = '#2ca02c'
# test_acc_color = '#ffdc78'

# data673_color
# val_auc_color = '#e377c2'
# test_auc_color = '#1f77b4'
# test_acc_color = '#ff6347'


# data673LR_color
val_auc_color = '#17becf' 
test_auc_color = '#ff5733'
test_acc_color = '#800080'

plt.figure(figsize=(18, 6))

# AUROC
plt.subplot(1, 3, 1)
plt.plot(range(1, 6), val_auc_folds, label='Validation AUROC', marker='o', color=val_auc_color)
plt.plot(range(1, 6), test_auc_folds, label='Test AUROC', marker='o', color=test_auc_color)
max_val_auc_idx = np.argmax(val_auc_folds) + 1
max_test_auc_idx = np.argmax(test_auc_folds) + 1
plt.scatter(max_val_auc_idx, val_auc_folds[max_val_auc_idx-1], color='black', zorder=5)
plt.text(max_val_auc_idx, val_auc_folds[max_val_auc_idx-1], f'{val_auc_folds[max_val_auc_idx-1]:.3f}',
         horizontalalignment='right', fontsize=12, color='black', verticalalignment='bottom')
plt.scatter(max_test_auc_idx, test_auc_folds[max_test_auc_idx-1], color='red', zorder=5)
plt.text(max_test_auc_idx, test_auc_folds[max_test_auc_idx-1], f'{test_auc_folds[max_test_auc_idx-1]:.3f}',
         horizontalalignment='right', fontsize=12, color='red', verticalalignment='bottom')
plt.title(f'AUROC per Fold\nMean Test AUROC: {mean_test_auc:.3f}', fontsize=14)
plt.xlabel('Fold', fontsize=14)
plt.ylabel('AUROC', fontsize=14)
plt.xticks(range(1, 6), fontsize=12)
plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], fontsize=12)
plt.legend(fontsize=12)

# AUPRC
plt.subplot(1, 3, 2)
plt.plot(range(1, 6), val_auprc_folds, label='Validation AUPRC', marker='o', color=val_auc_color)
plt.plot(range(1, 6), test_auprc_folds, label='Test AUPRC', marker='o', color=test_auc_color)
max_val_auprc_idx = np.argmax(val_auprc_folds) + 1
max_test_auprc_idx = np.argmax(test_auprc_folds) + 1
plt.scatter(max_val_auprc_idx, val_auprc_folds[max_val_auprc_idx-1], color='black', zorder=5)
plt.text(max_val_auprc_idx, val_auprc_folds[max_val_auprc_idx-1], f'{val_auprc_folds[max_val_auprc_idx-1]:.3f}',
         horizontalalignment='right', fontsize=12, color='black', verticalalignment='bottom')
plt.scatter(max_test_auprc_idx, test_auprc_folds[max_test_auprc_idx-1], color='red', zorder=5)
plt.text(max_test_auprc_idx, test_auprc_folds[max_test_auprc_idx-1], f'{test_auprc_folds[max_test_auprc_idx-1]:.3f}',
         horizontalalignment='right', fontsize=12, color='red', verticalalignment='bottom')
plt.title(f'AUPRC per Fold\nMean Test AUPRC: {mean_test_auprc:.3f}', fontsize=14)
plt.xlabel('Fold', fontsize=14)
plt.ylabel('AUPRC', fontsize=14)
plt.xticks(range(1, 6), fontsize=12)
plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], fontsize=12)
plt.legend(fontsize=12)

# Accuracy
plt.subplot(1, 3, 3)
plt.plot(range(1, 6), test_acc_folds, label='Test Accuracy', marker='o', color=test_acc_color)
max_test_acc_idx = np.argmax(test_acc_folds) + 1
plt.scatter(max_test_acc_idx, test_acc_folds[max_test_acc_idx-1], color='black', zorder=5)
plt.text(max_test_acc_idx, test_acc_folds[max_test_acc_idx-1], f'{test_acc_folds[max_test_acc_idx-1]:.3f}',
         horizontalalignment='right', fontsize=12, color='black', verticalalignment='bottom')
plt.title(f'Accuracy per Fold\nMean Test Accuracy: {mean_test_acc:.3f}', fontsize=14)
plt.xlabel('Fold', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.xticks(range(1, 6), fontsize=12)
plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], fontsize=12)
plt.legend(fontsize=12)

plt.tight_layout()
plt.savefig('/mnt/data0/users/baiy/CNV/figure/sup1_fig_673_LR_five_fold.pdf', dpi=300, bbox_inches='tight')
plt.show()