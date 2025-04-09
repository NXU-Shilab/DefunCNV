#!/bin/bash

# Get command line parameters
csv_file=$1
ref_fasta=$2
output_dir=$3
model_path=$4
cuda_flag=$5

# Generate output file path
alt_sequence="$output_dir/alt_sequence.fa"
ref_sequence="$output_dir/ref_sequence.fa"
alt_output="$output_dir/alt_sequence_predictions.h5"
ref_output="$output_dir/ref_sequence_predictions.h5"

# Making reference and variant sequences
echo "Running make_data.py to generate FASTA files..."
python3 make_data.py "$ref_fasta" "$csv_file" "$alt_sequence" "$ref_sequence"

if [ $? -ne 0 ]; then
    echo "make_data.py failed, stopping the execution."
    exit 1
fi

# Annotation.py was run to annotate the variant sequencesAnnotation.py was run to annotate the variant sequences
echo "Running annotation.py on alt_sequence.fa..."
python3 annotation.py --model_path "$model_path" \
                      --data_path "$alt_sequence" \
                      --output_path "$alt_output" \
                      $cuda_flag

if [ $? -ne 0 ]; then
    echo "annotation.py failed for alt_sequence, stopping the execution."
    exit 1
fi

# Annotation.py was run to annotate the reference sequence
echo "Running annotation.py on ref_sequence.fa..."
python3 annotation.py --model_path "$model_path" \
                      --data_path "$ref_sequence" \
                      --output_path "$ref_output" \
                      $cuda_flag

if [ $? -ne 0 ]; then
    echo "annotation.py failed for ref_sequence, stopping the execution."
    exit 1
fi

echo "Process completed. The results are saved in $output_dir."
echo "Generated files: $alt_output and $ref_output"
