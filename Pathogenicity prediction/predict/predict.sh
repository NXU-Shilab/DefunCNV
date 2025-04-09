#!/bin/bash

data_file=$1
model_path=$2
best_threshold_path=$3
output_file=$4

echo "Running predict.py..."
python3 predict.py --data_file "$data_file" --model_path "$model_path" --best_threshold_path "$best_threshold_path" --output_file "$output_file"

echo "Prediction completed. Results saved to $output_file."
