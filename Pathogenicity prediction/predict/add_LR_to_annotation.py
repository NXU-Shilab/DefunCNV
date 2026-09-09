import pandas as pd
import h5py
import numpy as np
import argparse


def add_LR_to_annotation(csv_file, h5_file, lr_column=6):
    csv_data = pd.read_csv(csv_file)

    # Resolve the LR score column: by integer index (default 6) or by name.
    if isinstance(lr_column, str):
        if lr_column not in csv_data.columns:
            raise ValueError(
                f"LR column '{lr_column}' not found. Available columns: {list(csv_data.columns)}")
        csv_column = csv_data[lr_column]
    else:
        if lr_column < 0 or lr_column >= csv_data.shape[1]:
            raise ValueError(
                f"LR column index {lr_column} out of range (0..{csv_data.shape[1] - 1}). "
                f"Available columns: {list(csv_data.columns)}")
        csv_column = csv_data.iloc[:, lr_column]

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
    parser.add_argument('--lr_column', type=str, default='LR_pred',
                        help="LR score column in the merged CSV: a column name (default LR_pred) "
                             "or an integer index, e.g. --lr_column 6")

    args = parser.parse_args()

    # Accept either an integer index or a column name.
    try:
        lr_column = int(args.lr_column)
    except ValueError:
        lr_column = args.lr_column

    add_LR_to_annotation(args.csv_file, args.h5_file, lr_column)
