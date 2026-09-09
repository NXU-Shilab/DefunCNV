#!/bin/bash
# DefunCNV feature annotation pipeline.
#
# Usage:
#   sh annotate.sh <csv_file> <ref_fasta> <output_dir> <model_path> [--cuda]
#
#   csv_file    : CNV input CSV. Required columns: Chromosome, Start, End, Type
#                (Type must be 'gain' or 'loss'). No 'chr' prefix on Chromosome.
#   ref_fasta   : hg19 or hg38 reference genome FASTA (headers must be 'chr'-prefixed,
#                e.g. >chr1). Download it from the Google Drive link in the README.
#   output_dir  : directory for the intermediate FASTA files and the output .h5 predictions.
#   model_path  : path to the trained DefunCNV model (best_model.pth.tar).
#   --cuda      : optional. Run on GPU if one is available; otherwise falls back to CPU.
#
# On Windows (Git Bash) where the interpreter is called `python` instead of `python3`,
# run:  PYTHON=python sh annotate.sh ...

PYTHON=${PYTHON:-python3}

# Get command line parameters
csv_file=$1
ref_fasta=$2
output_dir=$3
model_path=$4
cuda_flag=$5
# Strip any stray brackets that may have been copied from the docs,
# e.g. '[--cuda]' should be treated the same as '--cuda'.
cuda_flag=$(echo "$cuda_flag" | tr -d '[]')

if [ -z "$csv_file" ] || [ -z "$ref_fasta" ] || [ -z "$output_dir" ] || [ -z "$model_path" ]; then
    echo "Usage: sh annotate.sh <csv_file> <ref_fasta> <output_dir> <model_path> [--cuda]"
    exit 1
fi

# Generate output file path
mkdir -p "$output_dir"
alt_sequence="$output_dir/alt_sequence.fa"
ref_sequence="$output_dir/ref_sequence.fa"
alt_output="$output_dir/alt_sequence_predictions.h5"
ref_output="$output_dir/ref_sequence_predictions.h5"

# Making reference and variant sequences
echo "Running make_data.py to generate FASTA files..."
"$PYTHON" make_data.py "$ref_fasta" "$csv_file" "$alt_sequence" "$ref_sequence"

if [ $? -ne 0 ]; then
    echo "make_data.py failed, stopping the execution."
    exit 1
fi

# Annotation.py was run to annotate the variant sequences
echo "Running annotation.py on alt_sequence.fa..."
"$PYTHON" annotation.py --model_path "$model_path" \
                      --data_path "$alt_sequence" \
                      --output_path "$alt_output" \
                      $cuda_flag

if [ $? -ne 0 ]; then
    echo "annotation.py failed for alt_sequence, stopping the execution."
    exit 1
fi

# Annotation.py was run to annotate the reference sequence
echo "Running annotation.py on ref_sequence.fa..."
"$PYTHON" annotation.py --model_path "$model_path" \
                      --data_path "$ref_sequence" \
                      --output_path "$ref_output" \
                      $cuda_flag

if [ $? -ne 0 ]; then
    echo "annotation.py failed for ref_sequence, stopping the execution."
    exit 1
fi

echo "Process completed. The results are saved in $output_dir."
echo "Generated files: $alt_output and $ref_output"
