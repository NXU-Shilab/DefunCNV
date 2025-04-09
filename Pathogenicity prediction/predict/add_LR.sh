#!/bin/bash

# Get command line parameters
csv_file1=$1
csv_file2=$2
ref_fasta=$3
alt_fasta=$4
output_dir=$5
output_log2_file=$6
merged_csv=$7
cuda_flag=$8

# Calculate log2 score
echo "Running log2_scores.py..."
python3 log2_scores.py --ref_file "$ref_fasta" --alt_file "$alt_fasta" --output_file "$output_log2_file"

# Merge LR features to CSV file
echo "Running extract_LR_column.py..."
python3 extract_LR_column.py --file1 "$csv_file1" --file2 "$csv_file2" --output_file "$merged_csv"

echo "Running add_LR_to_annotation.py..."
python3 add_LR_to_annotation.py --csv_file "$merged_csv" --h5_file "$output_log2_file"

echo "Preprocessing completed. Files saved in $output_dir."
