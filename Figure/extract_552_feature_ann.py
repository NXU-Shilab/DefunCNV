import pandas as pd
import h5py
import numpy as np

csv_file_21907 = "/mnt/data0/users/baiy/CNV/data/feature/Sei_21907_feature.csv"
df_21907 = pd.read_csv(csv_file_21907)

csv_file_552 = "/mnt/data0/users/baiy/CNV/data/feature/552_feature.csv"
df_552 = pd.read_csv(csv_file_552)

features_21907 = df_21907['Feature'].values
features_552 = df_552['Feature'].values

feature_indices = []
for feature in features_552:
    if feature in features_21907:
        index = np.where(features_21907 == feature)[0][0]
        print(f"feature '{feature}' The corresponding index is {index}，The position in the 21907 feature is '{features_21907[index]}'")
        feature_indices.append(index)
        print(index)
    else:
        print(f"feature '{feature}' Not found in 21907 feature!")

h5_file = '/mnt/data0/users/baiy/Sei/sei-framework-main/train/retrained_sei_validate_data_12800/sei_validate_data_12800_label.h5'
with h5py.File(h5_file, 'r') as f:
    data_matrix = np.array(f['matrix'])

predictions = data_matrix[:, feature_indices]

new_h5_file = '/mnt/data0/users/baiy/Sei/sei-framework-main/train/retrained_sei_validate_data_12800/552_labels.h5'
with h5py.File(new_h5_file, 'w') as f_new:
    # f_new.create_dataset('predictions', data=predictions)
    f_new.create_dataset('matrix', data=predictions)

print("Successfully extracted and saved the new H5 file!")
