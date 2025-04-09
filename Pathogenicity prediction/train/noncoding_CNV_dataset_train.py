import random
import torch
import h5py
import numpy as np
import pandas as pd
import optuna
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

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

# Load data
def load_data(fold):

    with h5py.File(f"/mnt/data0/users/baiy/CNV/data/Brain_CNV_data/five_fold_dataset_0.956_best_model/673_LR_five_fold_data/train_data/fold_{fold + 1}_train_data.h5", 'r') as f_train:
        train_data = np.array(f_train['data'])
        train_labels = np.array(f_train['labels'])

    with h5py.File(f"/mnt/data0/users/baiy/CNV/data/Brain_CNV_data/five_fold_dataset_0.956_best_model/673_LR_five_fold_data/val_data/fold_{fold + 1}_val_data.h5", 'r') as f_val:
        val_data = np.array(f_val['data'])
        val_labels = np.array(f_val['labels'])

    return train_data, train_labels, val_data, val_labels

# Objective function of hyper parameter search
def objective(trial, fold):

    train_data, train_labels, val_data, val_labels = load_data(fold)

    n_estimators = trial.suggest_int('n_estimators', 50, 1000, step=50)
    # Building a random forest model
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=20)
    # Training model
    model.fit(train_data, train_labels)
    # Validation model
    val_preds = model.predict_proba(val_data)[:, 1]
    val_roc_auc = roc_auc_score(val_labels, val_preds)

    return val_roc_auc

all_fold_results = []

# Parameter search of every fold
for fold in range(5):

    # When creating optuna's study object, specify the sampler and fix the random seed
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=20))

    print(f'========== Fold {fold + 1} ==========')
    study.optimize(lambda trial: objective(trial, fold), n_trials=200)
    # Obtain the best parameters and retrain the model to save
    best_params = study.best_params
    train_data, train_labels, val_data, val_labels = load_data(fold)
    model = RandomForestClassifier(n_estimators=best_params['n_estimators'], random_state=20)
    model.fit(train_data, train_labels)
    val_preds = model.predict_proba(val_data)[:, 1]
    best_val_auc = roc_auc_score(val_labels, val_preds)
    # Save best model
    model_filename = f"/mnt/data0/users/baiy/CNV/data/Brain_CNV_data/five_fold_dataset_0.956_best_model/673_LR_five_fold_data/five_fold_best_param/rf_best_models/fold_{fold + 1}_best_model_rf.pkl"
    pd.to_pickle(model, model_filename)

    # Save best parameters to CSV
    best_config = {
        "fold": fold + 1,
        "n_estimators": best_params['n_estimators'],
        "best_auc": study.best_value
    }

    all_fold_results.append(best_config)

df_results = pd.DataFrame(all_fold_results)
df_results.to_csv(
    '/mnt/data0/users/baiy/CNV/data/Brain_CNV_data/five_fold_dataset_0.956_best_model/673_LR_five_fold_data/five_fold_best_param/fold_results_rf.csv',
    index=False)

print("The hyper parameter search and result saving are completed.")
