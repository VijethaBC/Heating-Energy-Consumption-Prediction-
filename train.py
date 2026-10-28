import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
import pickle
from datetime import datetime
import logging
from typing import Dict, Any

from data_loader import DataLoader
from models import ModelFactory

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> Dict[str, float]:
    """Calculate and return evaluation metrics"""
    metrics = {
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'r2': float(r2_score(y_true, y_pred))
    }
    
    logger.info(f"\n{model_name} Performance:")
    for metric, value in metrics.items():
        logger.info(f"{metric.upper()}: {value:.4f}")
    
    return metrics

def save_model(model: Any, model_name: str, metrics: Dict[str, float], output_dir: str):
    """Save the model and its metrics"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(output_dir, f"{model_name}.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Save metrics
    metrics_path = os.path.join(output_dir, f"{model_name}_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    logger.info(f"Model and metrics saved in {output_dir}")

def main():
    # Create output directory for this run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join('models', timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up file logging
    file_handler = logging.FileHandler(os.path.join(output_dir, 'training.log'))
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info("Starting model training pipeline")
    
    # Initialize data loader
    data_loader = DataLoader()
    
    # List of datasets to use
    datasets = ["uci_energy"]  # Using only UCI Energy dataset
    logger.info(f"Using datasets: {datasets}")
    
    # Parameters
    params = {
        'target_column': 'Appliances',  # Updated target column
        'sequence_length': 24,  # For LSTM/CNN models, using 24 hours of data
        'test_size': 0.2,
        'random_state': 42,
        'lstm_epochs': 30,  # Reduced from 100
        'batch_size': 64,  # Increased from 32
        'validation_split': 0.2
    }
    
    # Save parameters
    with open(os.path.join(output_dir, 'params.json'), 'w') as f:
        json.dump(params, f, indent=4)
    
    try:
        # Load and preprocess data for traditional models
        logger.info("Loading and preprocessing data...")
        X_scaled, y = data_loader.load_and_preprocess(
            dataset_names=datasets,
            target_column=params['target_column']
        )
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, 
            test_size=params['test_size'], 
            random_state=params['random_state']
        )
        
        logger.info(f"Data split - Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
        
        # Initialize model factory
        model_factory = ModelFactory()
        
        # Train and evaluate Random Forest
        logger.info("Training Random Forest model...")
        logger.info("Step 1/3: Creating Random Forest model...")
        rf_model = model_factory.create_random_forest()
        logger.info("Step 2/3: Tuning Random Forest hyperparameters...")
        rf_model = model_factory.tune_random_forest(rf_model, X_train, y_train)
        logger.info("Step 3/3: Making predictions...")
        rf_preds = rf_model.predict(X_test)
        rf_metrics = evaluate_model(y_test, rf_preds, "Random Forest")
        save_model(rf_model, "random_forest", rf_metrics, output_dir)
        
        # Train and evaluate SVR
        logger.info("Training SVR model...")
        svr_model = model_factory.create_svr()
        svr_model = model_factory.tune_svr(svr_model, X_train, y_train)
        svr_preds = svr_model.predict(X_test)
        svr_metrics = evaluate_model(y_test, svr_preds, "SVR")
        save_model(svr_model, "svr", svr_metrics, output_dir)
        
        # Load and preprocess data for sequential models
        logger.info("Preparing sequential data...")
        X_seq, y_seq = data_loader.load_and_preprocess(
            dataset_names=datasets,
            target_column=params['target_column'],
            sequence_length=params['sequence_length']
        )
        
        # Split sequential data
        X_train_seq, X_test_seq, y_train_seq, y_test_seq = train_test_split(
            X_seq, y_seq, 
            test_size=params['test_size'], 
            random_state=params['random_state']
        )
        
        # Train and evaluate LSTM
        logger.info("Training LSTM model...")
        lstm_model = model_factory.create_lstm(
            input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])
        )
        lstm_history = lstm_model.fit(
            X_train_seq, y_train_seq,
            epochs=params['lstm_epochs'],
            batch_size=params['batch_size'],
            validation_split=params['validation_split'],
            callbacks=model_factory.get_callbacks(),
            verbose=1
        )
        lstm_preds = lstm_model.predict(X_test_seq).flatten()
        lstm_metrics = evaluate_model(y_test_seq, lstm_preds, "LSTM")
        lstm_model.save(os.path.join(output_dir, 'lstm_model.keras'))
        
        # Save LSTM training history
        with open(os.path.join(output_dir, 'lstm_history.json'), 'w') as f:
            json.dump(lstm_history.history, f, indent=4)
        
        # Train and evaluate CNN-LSTM
        logger.info("Training CNN-LSTM model...")
        cnn_lstm_model = model_factory.create_cnn_lstm(
            input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])
        )
        cnn_lstm_history = cnn_lstm_model.fit(
            X_train_seq, y_train_seq,
            epochs=params['lstm_epochs'],
            batch_size=params['batch_size'],
            validation_split=params['validation_split'],
            callbacks=model_factory.get_callbacks(),
            verbose=1
        )
        cnn_lstm_preds = cnn_lstm_model.predict(X_test_seq).flatten()
        cnn_lstm_metrics = evaluate_model(y_test_seq, cnn_lstm_preds, "CNN-LSTM")
        cnn_lstm_model.save(os.path.join(output_dir, 'cnn_lstm_model.keras'))
        
        # Save CNN-LSTM training history
        with open(os.path.join(output_dir, 'cnn_lstm_history.json'), 'w') as f:
            json.dump(cnn_lstm_history.history, f, indent=4)
            
        logger.info("Training pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 