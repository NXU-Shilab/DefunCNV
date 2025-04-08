import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

df1 = pd.read_csv("/mnt/data0/users/baiy/Sei/sei-framework-main/train/cnn_with_swin_validate_data_12800/cnn_with_swin_533_feature_metrics_results_12800_0.956_best_model_del_19_nan.csv")
df1['AUC'] = pd.to_numeric(df1['AUC'], errors='coerce')

def categorize_feature(feature):
    parts = feature.split(' | ')
    if len(parts) > 1:
        if 'DNase' in parts[1]:
            return 'DNase'
        elif 'H' in parts[1] and 'K' in parts[1]:
            return 'Histone'
    return 'TF'

df1['Category'] = df1['Feature'].apply(categorize_feature)
df1['Source'] = 'cnn_with_swin'

plt.figure(figsize=(10, 6))
sns.violinplot(x='Category', y='AUC', data=df1, inner="quart", density_norm='count', palette="pastel")

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.title('AUROC distribution by feature category', fontsize=14)
plt.xlabel('Feature category', fontsize=14)
plt.ylabel('AUROC', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.savefig('/mnt/data0/users/baiy/CNV/figure/fig_2b_violin_plot.pdf', dpi=300, format='pdf', bbox_inches='tight')
plt.show()
