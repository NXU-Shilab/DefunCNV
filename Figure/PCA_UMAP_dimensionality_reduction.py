import h5py
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

features_names = pd.read_csv("/mnt/data0/users/baiy/CNV/data/feature/553_feature.csv")['Feature'].tolist()

h5_file = "/mnt/data0/users/baiy/CNV/data/h5_file/case_data/GBM/RecCNV_GBM_unique_noncoding_variants_2161_positive_pred_877.h5"


with h5py.File(h5_file, 'r') as f:

    features_data = f['positive_variants'][:]

adata = sc.AnnData(features_data)

adata.var['feature_names'] = features_names

sc.pp.neighbors(adata, n_neighbors=5)
sc.tl.diffmap(adata, n_comps=10)

sc.pp.neighbors(adata, n_pcs=10, use_rep='X_diffmap')

sc.tl.umap(adata)

sc.tl.leiden(adata, resolution=0.1)
print(adata.obs['leiden'].value_counts())

fig = sc.pl.umap(adata, color='leiden', title='Clustering of LGG positive variants', size=300, return_fig=True)

fig.savefig("/mnt/data0/users/baiy/CNV/figure/fig_5g_LGG_positive_pred_cluster_figure.pdf", dpi=300, format='pdf', bbox_inches='tight')

plt.show()

cluster_importance = {}


def clean_feature_name(feature_name):

    parts = feature_name.split('|')[:2]

    return ' | '.join(parts).strip()


for cluster in adata.obs['leiden'].cat.categories:

    cluster_data = adata.X[adata.obs['leiden'] == cluster]

    cluster_mean = cluster_data.mean(axis=0).flatten()

    cluster_feature_importance = pd.DataFrame({
        'Feature': adata.var['feature_names'],
        'Mean Value': cluster_mean,
        'Cluster': [cluster] * len(cluster_mean)
    })

    # Sort according to the mean to find the most important feature for the cluster
    cluster_feature_importance_sorted = cluster_feature_importance.sort_values(by='Mean Value', ascending=False)

    # Keep only the top 10 most important features and clean up the feature names
    cluster_importance[cluster] = [clean_feature_name(f) for f in
                                   cluster_feature_importance_sorted.head(10)['Feature'].tolist()]

cluster_names = []

for cluster, features in cluster_importance.items():
    cluster_names.append(features)

cluster_matrix = pd.DataFrame(cluster_names).T

print(cluster_matrix)

row_labels = [f'Feature {i + 1}' for i in range(cluster_matrix.shape[0])]

col_labels = [f'Cluster {i}' for i in range(cluster_matrix.shape[1])]

organisms_count = {}
features_count = {}

for cluster, features in cluster_importance.items():
    organisms_count[cluster] = {}
    features_count[cluster] = {}

    for feature in features:
        organism, feature_type = feature.split(' | ')

        # Count the number of times the organization name appears
        if organism not in organisms_count[cluster]:
            organisms_count[cluster][organism] = 0
        organisms_count[cluster][organism] += 1

        # Count the number of occurrences of feature categories
        if feature_type not in features_count[cluster]:
            features_count[cluster][feature_type] = 0
        features_count[cluster][feature_type] += 1

organisms_df = pd.DataFrame.from_dict(organisms_count).fillna(0).astype(int)

features_df = pd.DataFrame.from_dict(features_count).fillna(0).astype(int)

fig, ax = plt.subplots(1, 2, figsize=(18, 6))

or_heatmap = sns.heatmap(organisms_df, annot=True, fmt="d", cmap="YlGnBu", ax=ax[0], cbar_kws={'label': 'Count'})

or_bar = or_heatmap.collections[0].colorbar

or_bar.set_label('Count', fontsize=12)

ax[0].set_title('Organism frequency per cluster',fontsize=12)
ax[0].set_xlabel('Cluster',fontsize=12)
ax[0].set_ylabel('Organism',fontsize=12)

fe_heatmap = sns.heatmap(features_df, annot=True, fmt="d", cmap="YlGnBu", ax=ax[1], cbar_kws={'label': 'Count'})

fe_bar = fe_heatmap.collections[0].colorbar

fe_bar.set_label('Count', fontsize=12)


ax[1].set_title('Feature Frequency per Cluster',fontsize=12)
ax[1].set_xlabel('Cluster',fontsize=12)
ax[1].set_ylabel('Feature type',fontsize=12)

plt.tight_layout()

plt.savefig("/mnt/data0/users/baiy/CNV/figure/fig_5g_GBM_positive_pred_heatmap_figure.pdf", dpi=300, format='pdf', bbox_inches='tight')
plt.show()

# GBM YlGnBu
# LGG Blues