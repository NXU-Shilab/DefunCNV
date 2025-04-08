import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

file1 = "/mnt/data0/users/baiy/CNV/data/Brain_CNV_data/case_LGG/model_predictions_agree.csv"  # The two models predicted consistent categories
file2 = "/mnt/data0/users/baiy/CNV/data/Brain_CNV_data/case_LGG/2638LRrf_1_673LRrf_0.csv" # Model 1=1, model 2=0
file3 = "/mnt/data0/users/baiy/CNV/data/Brain_CNV_data/case_LGG/2638LRrf_0_673LRrf_1.csv" # Model 1=0, model 2=1

data1 = pd.read_csv(file1)
data2 = pd.read_csv(file2)
data3 = pd.read_csv(file3)

def plot_combined_confusion_matrix(data1, data2, data3):

    cm1 = confusion_matrix(data1.iloc[:, 9], data1.iloc[:, 9])  # The situation that the two models predict consistently
    cm2 = confusion_matrix(data2.iloc[:, 7], data2.iloc[:, 10])  # When model 1 predicts 1 and model 2 predicts 0
    cm3 = confusion_matrix(data3.iloc[:, 7], data3.iloc[:, 10])  # When model 1 predicts 0 and model 2 predicts 1

    combined_cm = [[0, 0], [0, 0]]

    combined_cm[0][0] = cm1[0][0]
    combined_cm[1][1] = cm1[1][1]
    combined_cm[1][0] = cm2[1][0]
    combined_cm[0][1] = cm3[0][1]

    plt.figure(figsize=(6, 5))
    sns.heatmap(combined_cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1],
                annot_kws={'size': 12})
    plt.title("LGG", fontsize=14)
    plt.xlabel('Noncoding DefunCNV predictions', fontsize=14)
    plt.ylabel('Coding/noncoding DefunCNV predictions', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig('/mnt/data0/users/baiy/CNV/figure/fig_5d_LGG_confusion_matrix.pdf', dpi=300,
                bbox_inches='tight', transparent=True)
    plt.show()

plot_combined_confusion_matrix(data1, data2, data3)

