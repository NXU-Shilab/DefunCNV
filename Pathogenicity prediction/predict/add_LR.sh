#!/bin/bash
# Add the XCNV LR score to the DefunCNV feature-annotation log2 scores.
#
# Usage (matches the README):
#   sh add_LR.sh <ref_annotation> <alt_annotation> <csv_file1> <csv_file2> <output_dir>
#
#   ref_annotation : ref_sequence_predictions.h5  (output of annotate.sh)
#   alt_annotation : alt_sequence_predictions.h5  (output of annotate.sh)
#   csv_file1      : the original CNV CSV you passed to annotate.sh
#   csv_file2      : XCNV LR-score CSV (https://github.com/kbvstmd/XCNV)
#   output_dir     : directory for intermediate files and the final .h5
#
# Column-name differences between the two CSVs (e.g. 'Chromosome' vs 'Chr',
# or a non-default LR column) are handled automatically by extract_LR_column.py.
# You may optionally override the LR column name as the 6th argument, e.g.:
#   sh add_LR.sh ref alt csv1 csv2 out LR
#
# The combined scores (log2 scores + LR column) are written to:
#   <output_dir>/log2_scores.h5
# which is then the <h5_file> argument for predict.sh.
#
# On Windows (Git Bash) run:  PYTHON=python sh add_LR.sh ...

PYTHON=${PYTHON:-python3}

# Get command line parameters (README order)
ref_annotation=$1
alt_annotation=$2
cnv_csv=$3
xcv_lr_csv=$4
output_dir=$5
lr_column=${6:-}   # optional override of the LR column name (default: LR_pred)

if [ -z "$ref_annotation" ] || [ -z "$alt_annotation" ] || [ -z "$cnv_csv" ] || [ -z "$xcv_lr_csv" ] || [ -z "$output_dir" ]; then
    echo "Usage: sh add_LR.sh <ref_annotation> <alt_annotation> <csv_file1> <csv_file2> <output_dir> [lr_column]"
    exit 1
fi

mkdir -p "$output_dir"

log2_file="$output_dir/log2_scores.h5"
merged_csv="$output_dir/cnv_with_LR.csv"

# Calculate log2 score from the ref/alt feature annotations
echo "Running log2_scores.py..."
"$PYTHON" log2_scores.py --ref_file "$ref_annotation" --alt_file "$alt_annotation" --output_file "$log2_file"

# Merge the XCNV LR score into the CNV CSV (column-name differences auto-detected)
echo "Running extract_LR_column.py..."
"$PYTHON" extract_LR_column.py --file1 "$cnv_csv" --file2 "$xcv_lr_csv" --output_file "$merged_csv" ${lr_column:+--lr_column "$lr_column"}

# Append the LR column to the HDF5 log2 scores (last column)
echo "Running add_LR_to_annotation.py..."
"$PYTHON" add_LR_to_annotation.py --csv_file "$merged_csv" --h5_file "$log2_file" ${lr_column:+--lr_column "$lr_column"}

echo "Preprocessing completed. The combined log2+LR scores are saved to $log2_file."
