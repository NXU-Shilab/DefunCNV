import h5py
import numpy as np
import pandas as pd
import joblib
import argparse

def predict(data_file, model_path, best_threshold_path, output_file):
    
    with h5py.File(data_file, 'r') as f:
        data = f['log2_scores'][:]

    data = np.asarray(data, dtype=np.float32)

    # Load optimal threshold
    best_threshold = np.load(best_threshold_path)
    print(f"Best threshold loaded: {best_threshold}")

    # Load the trained model
    model = joblib.load(model_path)

    # Obtain the predicted probability of the model
    predictions = model.predict_proba(data)[:, 1]

    # The best threshold was used to classify the prediction results
    predictions_labels = (predictions >= best_threshold).astype(int)

    # Save predict results as CSV files
    predictions_df = pd.DataFrame(predictions_labels, columns=['Predicted_Label'])
    predictions_df['Predicted_Score'] = predictions.round(3)
    predictions_df['Best_Threshold'] = best_threshold

    predictions_df.to_csv(output_file, index=False)

    print(f"Predictions saved to {output_file}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict pathogenicity using pre-trained model")
    parser.add_argument('--data_file', type=str, required=True, help="Path to the input data (log2 scores HDF5 file)")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained model file")
    parser.add_argument('--best_threshold_path', type=str, required=True, help="Path to the best threshold file")
    parser.add_argument('--output_file', type=str, required=True, help="Path to save predictions CSV file")

    args = parser.parse_args()
    
    # predict
    predict(args.data_file, args.model_path, args.best_threshold_path, args.output_file)
