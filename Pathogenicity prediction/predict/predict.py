import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
import joblib  

# Load new data for prediction
data_file = "/mnt/data0/users/baiy/CNV/data/Brain_CNV_data/review_case/review_case_hg38_3_log2_scores_LR.h5"

with h5py.File(data_file, 'r') as f:
    data = f['log2_scores'][:]  # Replace 'data' with the correct dataset name in the HDF5 file

# Ensure that the data is a NumPy array
data = np.asarray(data, dtype=np.float32)

# Load the best model 
best_model_path = "/mnt/data0/users/baiy/CNV/data/Brain_CNV_data/five_fold_dataset_0.956_best_model/673_LR_five_fold_data/five_fold_best_param/rf_best_models/fold_2_best_model_rf.pkl"

# Load the best threshold from file
best_threshold_path = "/mnt/data0/users/baiy/CNV/data/Brain_CNV_data/five_fold_dataset_0.956_best_model/673_LR_five_fold_data/five_fold_best_param/best_threshold_fold_2.npy"
best_threshold = np.load(best_threshold_path)
print(best_threshold)

# Load the pre-trained Random Forest model
model = joblib.load(best_model_path) 

predictions = model.predict_proba(data)[:, 1]

# Use the best threshold to classify predictions
predictions_labels = (predictions >= best_threshold).astype(int)

# Save predictions to a CSV file
predictions_df = pd.DataFrame(predictions_labels, columns=['Predicted_Label']) 
predictions_df['Predicted_Score'] = predictions.round(3)
# Add the best_threshold to the dataframe (it's a constant for all rows)
predictions_df['Best_Threshold'] = best_threshold
predictions_df.to_csv("/mnt/data0/users/baiy/CNV/data/Brain_CNV_data/review_case/673LR_rf_best_model_review_case_hg38_3_predictions.csv", index=False)

print("Predictions saved to csv.")
