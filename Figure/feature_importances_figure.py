import h5py
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

features_file = "/mnt/data0/users/baiy/CNV/data/h5_file/train_data_0.956_best_model/2638_data/Clinvar_30_brain_dieases_cnv_HG38_2638_train_log2_scores_LR.h5"
labels_file = "/mnt/data0/users/baiy/CNV/data/h5_file/train_data_0.956_best_model/2638_data/Clinvar_30_brain_dieases_cnv_HG38_2638_labels.h5"
# feature_names_file = "/mnt/data0/users/baiy/CNV/data/feature/552_feature.csv"  # 552
feature_names_file = "/mnt/data0/users/baiy/CNV/data/feature/553_feature.csv"  # 553

with h5py.File(features_file, 'r') as f:
    data = np.asarray(f['log2_scores'], dtype=np.float32)

with h5py.File(labels_file, 'r') as f:
    labels = np.asarray(f['Labels'], dtype=np.float32)

print(data.shape)
print(labels.shape)

feature_names_df = pd.read_csv(feature_names_file)
feature_names = feature_names_df.iloc[:, 0].values

# Extract all parts before the second '|'
feature_names = [name.split('|')[:2] for name in feature_names]
feature_names = [' | '.join(parts).strip() for parts in feature_names]

assert len(feature_names) == data.shape[1], "The number of feature names does not match the number of data columns"

X = pd.DataFrame(data, columns=feature_names)

best_model_path = "/mnt/data0/users/baiy/CNV/data/Brain_CNV_data/five_fold_dataset_0.956_best_model/2638_LR_five_fold_data/five_fold_best_param/rf_best_models/fold_2_best_model_rf.pkl"
best_model = joblib.load(best_model_path)

# Get feature importance
importances = best_model.feature_importances_

# Ranking feature importance from large to small
indices = importances.argsort()[::-1]

# Select the top 20 features
top_n = 20
top_indices = indices[:top_n]
top_importances = importances[top_indices]

# Get the names of the top 20 features
top_feature_names = [X.columns[i] for i in top_indices]

print("Top 20 feature importance:")
for i in range(top_n):
    print(f"Feature {top_feature_names[i]}: {top_importances[i]}")

plt.figure(figsize=(18, 8))
# plt.title("Top 20 feature importance of DefunCNV in noncoding dataset", fontsize=14)
plt.title("Top 20 feature importance of DefunCNV in coding/noncoding dataset with LR feature added", fontsize=14)
plt.barh(range(top_n), top_importances, align="center",color='#ADD8E6')

plt.yticks(range(top_n), top_feature_names)
plt.gca().invert_yaxis()
plt.xlabel("Importance", fontsize=14)



# plt.xticks(rotation=45)
plt.yticks(rotation=30)
plt.xticks(fontsize=12)
plt.yticks(fontsize=8)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.savefig('/mnt/data0/users/baiy/CNV/figure/sup3_2638_LR_feature_importance.pdf', dpi=300, bbox_inches='tight')
plt.show()

# Light grey - #D3D3D3
# Light blue - #ADD8E6
# Light orange - #FFDAB9
# Light purple - #E6E6FA
