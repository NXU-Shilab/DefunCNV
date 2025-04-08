import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

defun_path = "/mnt/data0/users/baiy/Sei/sei-framework-main/train/cnn_with_swin_validate_data_12800/cnn_with_swin_533_feature_metrics_results_12800_0.956_best_model_del_19_nan.csv"
sei_path = "/mnt/data0/users/baiy/Sei/sei-framework-main/train/retrained_sei_validate_data_12800/retrained_sei_533_feature_metrics_results_12800_new_del_19_nan.csv"

defun_cnv_df = pd.read_csv(defun_path)
sei_df = pd.read_csv(sei_path)

defun_cnv_auc = defun_cnv_df.iloc[:, 1].values  # Obtain AUC value of DefunCNV model
sei_auc = sei_df.iloc[:, 1].values  # Obtain the AUC value of SEI model

auc_data = pd.DataFrame({
    'DefunCNV AUC': defun_cnv_auc,
    'Sei AUC': sei_auc
})

defun_cnv_avg_auc = auc_data['DefunCNV AUC'].mean()
sei_avg_auc = auc_data['Sei AUC'].mean()

plt.rcParams.update({
    'font.size': 12
})
fig = plt.figure(figsize=(6, 6))

gs = gridspec.GridSpec(2, 2,
                      height_ratios=[1, 8],
                      width_ratios=[8, 1],
                      hspace=0.05, wspace=0.05)

ax_main = fig.add_subplot(gs[1, 0])

ax_main.scatter(
    x=auc_data['DefunCNV AUC'],
    y=auc_data['Sei AUC'],
    s=100,
    alpha=0.7,
    color="#2c7bb6",
    edgecolor="white",
    linewidth=0.5
)

ax_main.set_xlabel("DefunCNV AUROC per feature", fontsize=12)
ax_main.set_ylabel("Sei AUROC per feature", fontsize=12)
ax_main.set(xlim=(0, 1.02), ylim=(0, 1.02))
ax_main.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax_main.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax_main.tick_params(axis='both', which='major', length=6)
ax_main.spines['top'].set_visible(False)
ax_main.spines['right'].set_visible(False)

ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
# ax_top.set_xticks([])
# ax_top.set_yticks([])
sns.histplot(
    auc_data['DefunCNV AUC'],
    bins=25,
    color="#2c7bb6",
    alpha=0.8,
    ax=ax_top
)
ax_top.text(0.3, 90,
          f"Mean AUROC = {auc_data['DefunCNV AUC'].mean():.3f}",
          fontsize=12)

ax_top.tick_params(axis='x', which='both',
                  bottom=True,
                  labelbottom=False)
ax_top.tick_params(axis='y', which='both',
                  left=False,
                  labelleft=False)

ax_top.set_xlabel('')
ax_top.set_ylabel('')
ax_top.spines['top'].set_visible(False)
ax_top.spines['right'].set_visible(False)
ax_top.spines['left'].set_visible(False)
ax_top.spines['bottom'].set_visible(True)

ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)
sns.histplot(
    y=auc_data['Sei AUC'],
    bins=25,
    color="#d7191c",
    alpha=0.8,
    ax=ax_right,
    orientation='horizontal'
)
ax_right.text(110, 0.5,
          f"Mean AUROC = {auc_data['Sei AUC'].mean():.3f}",
          fontsize=12,
          va ='center',
          ha ='left',
          rotation = 270
)

ax_right.tick_params(axis='x', which='both',
                   bottom=False,
                   labelbottom=False)
ax_right.tick_params(axis='y', which='both',
                   left=True,
                   labelleft=False)

ax_right.set_xlabel('')
ax_right.set_ylabel('')
ax_right.spines['top'].set_visible(False)
ax_right.spines['right'].set_visible(False)
ax_right.spines['left'].set_visible(True)
ax_right.spines['bottom'].set_visible(False)

plt.savefig('/mnt/data0/users/baiy/CNV/figure/fig_2a_scatter_plot.pdf', dpi=300, format='pdf',bbox_inches='tight')
plt.show()