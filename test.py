import os
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from data_loader import DataLoader

def load_model(model_dir, model_name):
    """Load a trained model from the specified directory"""
    model_path = os.path.join(model_dir, f"{model_name}.pkl")
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    return None

def evaluate_predictions(y_true, y_pred, model_name):
    """Calculate and print evaluation metrics"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n{model_name} Performance:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    
    return rmse, mae, r2

def main():
    # Get the latest model directory
    model_dirs = [d for d in os.listdir('models') if os.path.isdir(os.path.join('models', d))]
    latest_model_dir = max(model_dirs)
    model_dir = os.path.join('models', latest_model_dir)
    print(f"Using models from: {model_dir}")
    
    # Initialize data loader
    data_loader = DataLoader()
    
    # Load and preprocess test data
    print("\nLoading and preprocessing test data...")
    X_scaled, y = data_loader.load_and_preprocess(
        dataset_names=['uci_energy'],
        target_column='Appliances'
    )
    
    # Split into train/test (to get the same test set as training)
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    # Test traditional models
    models = ['random_forest', 'svr']
    for model_name in models:
        model = load_model(model_dir, model_name)
        if model is not None:
            print(f"\nTesting {model_name}...")
            predictions = model.predict(X_test)
            evaluate_predictions(y_test, predictions, model_name.replace('_', ' ').title())
    
    # Prepare sequential data for LSTM/CNN-LSTM
    print("\nPreparing sequential data...")
    X_seq, y_seq = data_loader.load_and_preprocess(
        dataset_names=['uci_energy'],
        target_column='Appliances',
        sequence_length=24
    )
    
    # Split sequential data
    _, X_test_seq, _, y_test_seq = train_test_split(
        X_seq, y_seq, test_size=0.2, random_state=42
    )
    
    # Test LSTM and CNN-LSTM models
    for model_name in ['lstm', 'cnn_lstm']:
        model_path = os.path.join(model_dir, f"{model_name}_model.keras")
        if os.path.exists(model_path):
            print(f"\nTesting {model_name}...")
            model = tf.keras.models.load_model(model_path)
            predictions = model.predict(X_test_seq).flatten()
            evaluate_predictions(y_test_seq, predictions, model_name.replace('_', '-').upper())

if __name__ == "__main__":
    main() 