import h5py
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

feature_names_file = "/mnt/data0/users/baiy/CNV/data/feature/553_feature.csv"
features_names = pd.read_csv(feature_names_file)['Feature'].tolist()

h5_file = "/mnt/data0/users/baiy/CNV/data/Brain_CNV_data/review_case/review_case_hg38_3_positive_pred_2.h5"

with h5py.File(h5_file, 'r') as f:

    features_data = f['positive_variants'][:]

top_features_dict = {}

def clean_feature_name(feature_name):

    parts = feature_name.split('|')[:2]

    return ' | '.join(parts).strip()


for idx, variant_scores in enumerate(features_data):

    feature_importance = pd.DataFrame({
        'Feature': features_names,
        'Score': variant_scores
    })

    feature_importance['Feature'] = feature_importance['Feature'].apply(clean_feature_name)

    top_features = feature_importance.sort_values(by='Score', ascending=False).head(10)
    top_features_dict[idx] = top_features

fig, axes = plt.subplots(1, len(top_features_dict), figsize=(15, 6))

if len(top_features_dict) == 1:
    axes = [axes]

cmap_list = ["YlGnBu", "Blues"]
title_list = ["Top 10 features for variant chr22: 18970514-19031242 loss",
              "Top 10 features for variant chr2: 7918076-7977796 loss"]

for idx, variant_scores in enumerate(features_data):
    top_features = top_features_dict[idx]

    feature_matrix = np.array([top_features['Score']])

    cs_heatmap = sns.heatmap(feature_matrix, annot=True, cmap=cmap_list[idx], ax=axes[idx], cbar_kws={'label': 'Score'},
                             xticklabels=top_features['Feature'], yticklabels=['Score'], cbar=True)

    cs_bar = cs_heatmap.collections[0].colorbar
    cs_bar.set_label('Score', fontsize=12)

    axes[idx].set_title(title_list[idx], fontsize=12)

    axes[idx].set_yticks(np.arange(1))
    axes[idx].set_xticklabels(top_features['Feature'], rotation=45, ha="right", fontsize=10)
    axes[idx].set_xlabel('Feature', fontsize=12)


plt.savefig("/mnt/data0/users/baiy/CNV/figure/fig_6a_review_case_positive_pred_heatmap_figure.pdf", dpi=300, format='pdf', bbox_inches='tight')
plt.tight_layout()
plt.show()
