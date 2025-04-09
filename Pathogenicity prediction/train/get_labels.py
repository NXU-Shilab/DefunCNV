import h5py
import numpy as np
import pandas as pd
def get_labels(excel_file, output_file):

    df = pd.read_csv(excel_file)

    # Create a new column'label'and mark it according to the value of the' pathogenicity 'column
    # df['Label'] = df['Pathogenicity'].apply(lambda x: 0 if x == 'benign' else 1)
    df['Label'] = df['Pathogenicity'].apply(lambda x: 0 if x in ['Benign'] else 1)
    # df['Label'] = df['Pathogenicity'].apply(lambda x: 0 if x in ['Benign', 'Likely benign'] else 1)

    with h5py.File(output_file, 'w') as h5_file:
        # for column in df.columns:
        h5_file.create_dataset('Labels', data=df['Label'].values)

excel_file = '/mnt/data0/users/baiy/CNV/data/Brain_CNV_data/Clinvar_30_brain_dieases_cnv_HG38_noncoding_variants_146_nc527_673.csv'
output_file = '/mnt/data0/users/baiy/CNV/data/h5_file/train_data_0.956_best_model/673_data/Clinvar_30_brain_dieases_cnv_HG38_noncoding_variants_146_nc527_673_labels.h5'

get_labels(excel_file, output_file)