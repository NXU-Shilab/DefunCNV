import pandas as pd
import argparse

def extract_LR_column(file1, file2, output_file):
    
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    
    matching_columns = df1.columns[:4]
    df2_matching_columns = df2.columns[:4]

    
    df_merged = pd.merge(df1, df2[df2_matching_columns.tolist() + [df2.columns[12]]],
                         left_on=matching_columns.tolist(),
                         right_on=df2_matching_columns.tolist(),
                         how='left')

    df_merged = df_merged.drop_duplicates()

    df_merged.to_csv(output_file, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract LR score column from a second CSV and merge it with the first CSV")
    parser.add_argument('--file1', type=str, required=True, help="Path to the first CSV file")
    parser.add_argument('--file2', type=str, required=True, help="Path to the second CSV file")
    parser.add_argument('--output_file', type=str, required=True, help="Path to save the merged CSV file")

    args = parser.parse_args()
    
    extract_LR_column(args.file1, args.file2, args.output_file)
