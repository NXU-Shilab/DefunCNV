import pandas as pd
import h5py
import numpy as np
import argparse

def add_LR_to_annotation(csv_file, h5_file):
    csv_data = pd.read_csv(csv_file)
    csv_column = csv_data.iloc[:, 6]  # Get the LR score column of CSV file

    with h5py.File(h5_file, 'r+') as h5f:
        dataset_name = 'log2_scores'
        h5_data = h5f[dataset_name][:]

        if len(h5_data) != len(csv_column):
            raise ValueError("The length of LR score column in CSV file does not match the number of data set rows in HDF5 file!")

        # Add the LR score column of CSV as the new last column
        new_data = np.column_stack((h5_data, csv_column.values))

        del h5f[dataset_name]
        h5f.create_dataset(dataset_name, data=new_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add LR score column from CSV to the HDF5 annotation file")
    parser.add_argument('--csv_file', type=str, required=True, help="Path to the CSV file with LR scores")
    parser.add_argument('--h5_file', type=str, required=True, help="Path to the HDF5 annotation file")
    
    args = parser.parse_args()
    
    add_LR_to_annotation(args.csv_file, args.h5_file)
