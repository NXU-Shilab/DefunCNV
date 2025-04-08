import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

bed_file_path = "/mnt/data0/users/baiy/CNV/data/feature/552_feature_bed_file.bed"
bed_df = pd.read_csv(bed_file_path, sep='\t', header=None, names=['chrom', 'start', 'end', 'feature'])

bed_df['length_kb'] = (bed_df['end'] - bed_df['start']) / 1000

def categorize_feature(feature):
    parts = feature.split(' | ')
    if len(parts) > 1:
        second_part = parts[1]
        if 'DNase' in second_part:
            return 'DNase'
        elif second_part.startswith('H') and any(c.isdigit() for c in second_part):
            return 'Histone'
    return 'TF'

bed_df['feature_type'] = bed_df['feature'].apply(categorize_feature)

plt.figure(figsize=(10, 6))
sns.boxplot(
    x='feature_type',
    y='length_kb',
    data=bed_df,
    order=['TF', 'DNase', 'Histone'],
    showfliers=False,
    palette={'TF': 'skyblue', 'DNase': 'lightgreen', 'Histone': 'lightcoral'}
)

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.title('Feature length distribution by type', fontsize=14)
plt.xlabel('Feature type', fontsize=12)
plt.ylabel('Length (kb)', fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.savefig('/mnt/data0/users/baiy/CNV/figure/fig_2c_box_plot_feature_length.pdf', dpi=300, format='pdf', bbox_inches='tight')
plt.show()
