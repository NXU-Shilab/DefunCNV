import pandas as pd
import h5py
import numpy as np

csv_file = "/mnt/data0/users/baiy/CNV/data/Brain_CNV_data/review_case/review_case_hg38_3_contains_LR.csv"
csv_data = pd.read_csv(csv_file)
csv_column = csv_data.iloc[:, 6]  # Get the LR score column of CSV file

h5_file = "/mnt/data0/users/baiy/CNV/data/Brain_CNV_data/review_case/review_case_hg38_3_log2_scores_LR.h5"
with h5py.File(h5_file, 'r+') as h5f:

    dataset_name = 'log2_scores'
    h5_data = h5f[dataset_name][:]

    if len(h5_data) != len(csv_column):
        raise ValueError("The length of LR score column in CSV file does not match the number of data set rows in HDF5 file!")

    # Add the LR score column of CSV as the new last column
    new_data = np.column_stack((h5_data, csv_column.values))

    del h5f[dataset_name]
    h5f.create_dataset(dataset_name, data=new_data)

print("LR score column of CSV file has been successfully added to HDF5 file!")
