import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn import venn2

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

gbm_file = "/mnt/data0/users/baiy/CNV/data/Brain_CNV_data/case_GBM/RecCNV_GBM_4217.csv"
lgg_file = "/mnt/data0/users/baiy/CNV/data/Brain_CNV_data/case_LGG/RecCNV_LGG_881.csv"

gbm_data = pd.read_csv(gbm_file)
lgg_data = pd.read_csv(lgg_file)

gbm_variants = gbm_data.iloc[:, :4].apply(tuple, axis=1).tolist()
lgg_variants = lgg_data.iloc[:, :4].apply(tuple, axis=1).tolist()

gbm_set = set(gbm_variants)
lgg_set = set(lgg_variants)

common_variants = gbm_set.intersection(lgg_set)
gbm_unique = gbm_set.difference(lgg_set)
lgg_unique = lgg_set.difference(gbm_set)

total_gbm = len(gbm_set)
total_lgg = len(lgg_set)

difference_ratio = abs(total_gbm - total_lgg) / max(total_gbm, total_lgg)

scale_factor_gbm = 1 - (difference_ratio * 0.6)
scale_factor_lgg = 1 + (difference_ratio * 0.6)

plt.figure(figsize=(8, 6))
venn = venn2(subsets=(len(gbm_unique) * scale_factor_gbm, len(lgg_unique) * scale_factor_lgg, len(common_variants)),
             set_labels=('GBM', 'LGG'))

venn.get_label_by_id('10').set_text(f'{len(gbm_unique)}')
venn.get_label_by_id('01').set_text(f'{len(lgg_unique)}')
venn.get_label_by_id('11').set_text(f'{len(common_variants)}')

for label in venn.set_labels:
    label.set_fontsize(14)
    label.set_fontweight('bold')
    label.set_color('#1f77b4')

venn.get_patch_by_id('10').set_color('#cce5ff')
venn.get_patch_by_id('01').set_color('#ffebcc')
venn.get_patch_by_id('11').set_color('#ffb3b3')

venn.get_label_by_id('10').set_fontsize(12)
venn.get_label_by_id('01').set_fontsize(12)
venn.get_label_by_id('11').set_fontsize(12)

plt.tight_layout()
plt.savefig('/mnt/data0/users/baiy/CNV/figure/fig_5a_venn_plot_all_variants.pdf', dpi=300, bbox_inches='tight', transparent=True)

plt.show()


#all variants color
#cce5ff
#ffebcc
#ffb3b3

#noncoding variants color
#ffcc99
#b3e0ff
#ff6666