import pandas as pd

file1 = "/mnt/data0/users/baiy/CNV/data/Brain_CNV_data/review_case/review_case_hg38_3.csv"
df1 = pd.read_csv(file1)

file2 = "/mnt/data0/users/baiy/CNV/data/Brain_CNV_data/review_case/review_case_hg38_3.output.csv"
df2 = pd.read_csv(file2)

# Select the first four columns as matching columns
matching_columns = df1.columns[:4]
df2_matching_columns = df2.columns[:4]

# Match according to the first four columns, merge the two dataframes,
# and add the 13th column of the second CSV to the first CSV
df_merged = pd.merge(df1, df2[df2_matching_columns.tolist() + [df2.columns[12]]],
                     left_on=matching_columns.tolist(),
                     right_on=df2_matching_columns.tolist(),
                     how='left')

# Remove completely duplicate rows from the merged data
df_merged = df_merged.drop_duplicates()

# Save the merged data as a new CSV file
df_merged.to_csv("/mnt/data0/users/baiy/CNV/data/Brain_CNV_data/review_case/review_case_hg38_3_contains_LR.csv", index=False)
