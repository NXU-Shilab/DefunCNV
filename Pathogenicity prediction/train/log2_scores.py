import h5py
import numpy as np
import pandas as pd


def calculate_log2_scores(ref_file, alt_file, output_file):
    
    with h5py.File(ref_file, 'r') as ref_h5, h5py.File(alt_file, 'r') as alt_h5:
        
        ref_data = ref_h5['predictions'][:]
        alt_data = alt_h5['predictions'][:]

        # Calculate log2 score
        # Calculate absolute difference
        diff_abs = ref_data - alt_data
        # Calculate relative differences
        logit_diff = np.log2(ref_data / (1 - ref_data + 1e-12)) - np.log2(alt_data / (1 - alt_data + 1e-12))
        # Calculate the product of absolute difference and relative difference and take the absolute value
        log2_scores = np.abs(diff_abs * logit_diff)

        with h5py.File(output_file, 'w') as output_h5:
            output_h5.create_dataset('log2_scores_diff_abs', data=diff_abs)
            output_h5.create_dataset('log2_scores_logit_diff', data=logit_diff)
            output_h5.create_dataset('log2_scores', data=log2_scores)

ref_file = "/mnt/data0/users/baiy/CNV/data/Brain_CNV_data/review_case/review_case_hg38_3_ref_sequence_predictions.h5"
alt_file = "/mnt/data0/users/baiy/CNV/data/Brain_CNV_data/review_case/review_case_hg38_3_alt_sequence_predictions.h5"
output_file = "/mnt/data0/users/baiy/CNV/data/Brain_CNV_data/review_case/review_case_hg38_3_log2_scores_LR.h5"

calculate_log2_scores(ref_file, alt_file, output_file)

