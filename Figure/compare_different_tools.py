import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

data = pd.read_csv(
    "/mnt/data0/users/baiy/CNV/data/Brain_CNV_data/five_fold_dataset_0.956_best_model/three_models_comparation_on_four_dataset/673_LR_fold_2_compare.csv",
    index_col=0)

data_melted = data.reset_index().melt(id_vars='index', var_name='Metric', value_name='Value')
data_melted.columns = ['Tool', 'Metric', 'Value']

sns.set(style='ticks')
plt.figure(figsize=(10, 6))
bar_plot = sns.barplot(data=data_melted, x='Metric', y='Value', hue='Tool', palette=['#9392BE', '#D0E7ED', '#D5E4A8'])

bar_plot.set_xlabel("", fontsize=14)
bar_plot.set_ylabel("Value", fontsize=14)
bar_plot.set(ylim=(0, 1))

for p in bar_plot.patches:
    height = p.get_height()
    if height > 0.01:
        bar_plot.annotate(f'{height:.3f}', (p.get_x() + p.get_width() / 2., height),
                          ha='center', va='bottom', fontsize=10, color='black')

plt.legend(title='Tools', loc='upper left', frameon=True)

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.savefig('/mnt/data0/users/baiy/CNV/figure/sup2_673LR_three_tools_compar.pdf', dpi=300, bbox_inches='tight')

plt.tight_layout()
plt.show()

# data2638_color
# MLP = '#445681'
# X-CNV = '#2E827F'
# DefunCNV = '#6BBB6E'

# data2638LR_color
# MLP = '#DBEAF3'
# X-CNV = '#B6D7E9'
# DefunCNV = '#82A7D1'

# data673_color
# MLP = '#F7E0D3'
# X-CNV = '#EEA58E'
# DefunCNV = '#F2AB6A'


# data673LR_color
# MLP = '#9392BE'
# X-CNV = '#D0E7ED'
# DefunCNV = '#D5E4A8'