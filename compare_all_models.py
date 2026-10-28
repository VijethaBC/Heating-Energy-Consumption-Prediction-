import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time
import warnings
warnings.filterwarnings('ignore')

# Import the model comparison class for BiGRU, GRU, and TCN
from model_comparison import ModelComparison

# Import the model factory for LSTM, CNN-LSTM, Random Forest, and SVR
from models import ModelFactory

def generate_data(n_samples, sequence_length, n_features):
    """Generate synthetic time series data"""
    X = np.random.randn(n_samples, sequence_length, n_features)
    y = np.sum(X, axis=2).mean(axis=1, keepdims=True)
    return X, y

def evaluate_model(y_true, y_pred, model_name):
    """Calculate evaluation metrics"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'Model': model_name,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2
    }

def main():
    # Parameters
    sequence_length = 10
    n_features = 5
    n_outputs = 1
    n_samples = 1000
    test_size = 0.2
    random_state = 42
    epochs = 50  # Reduced for faster execution
    
    print("Generating synthetic data...")
    # Generate data
    X, y = generate_data(n_samples, sequence_length, n_features)
    
    # Split data for standard models
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Reshape for traditional models (Random Forest, SVR)
    X_train_2d = X_train.reshape(X_train.shape[0], -1)
    X_test_2d = X_test.reshape(X_test.shape[0], -1)
    
    # Results storage
    results = []
    training_times = []
    
    # Part 1: Models from ModelComparison (BiGRU, GRU, TCN)
    print("\nTraining BiGRU, GRU, and TCN models...")
    model_comparison = ModelComparison(sequence_length, n_features, n_outputs)
    
    start_time = time.time()
    mc_results = model_comparison.train_and_evaluate(
        X_train, y_train, X_test, y_test, epochs=epochs
    )
    end_time = time.time()
    
    for model_name, metrics in mc_results.items():
        results.append({
            'Model': model_name,
            'RMSE': np.sqrt(metrics['MSE']),
            'MAE': metrics['MAE'],
            'R²': metrics['R2']
        })
        training_times.append({
            'Model': model_name,
            'Training Time (s)': (end_time - start_time) / 3  # Approximation for each model (updated for 3 models)
        })
    
    # Part 2: Models from ModelFactory
    model_factory = ModelFactory()
    
    # Random Forest
    print("Training Random Forest model...")
    start_time = time.time()
    rf_model = model_factory.create_random_forest()
    rf_model.fit(X_train_2d, y_train.ravel())
    rf_preds = rf_model.predict(X_test_2d)
    end_time = time.time()
    
    results.append(evaluate_model(y_test, rf_preds.reshape(-1, 1), "Random Forest"))
    training_times.append({
        'Model': "Random Forest",
        'Training Time (s)': end_time - start_time
    })
    
    # SVR
    print("Training SVR model...")
    start_time = time.time()
    svr_model = model_factory.create_svr()
    svr_model.fit(X_train_2d, y_train.ravel())
    svr_preds = svr_model.predict(X_test_2d)
    end_time = time.time()
    
    results.append(evaluate_model(y_test, svr_preds.reshape(-1, 1), "SVR"))
    training_times.append({
        'Model': "SVR",
        'Training Time (s)': end_time - start_time
    })
    
    try:
        # LSTM
        print("Training LSTM model...")
        start_time = time.time()
        lstm_model = model_factory.create_lstm(
            input_shape=(X_train.shape[1], X_train.shape[2])
        )
        lstm_model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )
        lstm_preds = lstm_model.predict(X_test)
        end_time = time.time()
        
        results.append(evaluate_model(y_test, lstm_preds, "LSTM"))
        training_times.append({
            'Model': "LSTM",
            'Training Time (s)': end_time - start_time
        })
        
        # CNN-LSTM
        print("Training CNN-LSTM model...")
        start_time = time.time()
        cnn_lstm_model = model_factory.create_cnn_lstm(
            input_shape=(X_train.shape[1], X_train.shape[2])
        )
        cnn_lstm_model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )
        cnn_lstm_preds = cnn_lstm_model.predict(X_test)
        end_time = time.time()
        
        results.append(evaluate_model(y_test, cnn_lstm_preds, "CNN-LSTM"))
        training_times.append({
            'Model': "CNN-LSTM",
            'Training Time (s)': end_time - start_time
        })
    except ImportError:
        print("TensorFlow not properly installed. Skipping LSTM and CNN-LSTM models.")
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    training_times_df = pd.DataFrame(training_times)
    
    # Print results
    print("\n===== MODEL COMPARISON RESULTS =====")
    print(results_df.to_string(index=False))
    
    print("\n===== TRAINING TIMES =====")
    print(training_times_df.to_string(index=False))
    
    # Plot results
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # RMSE plot (Fig1)
    axs[0, 0].bar(results_df['Model'], results_df['RMSE'])
    axs[0, 0].set_title('RMSE (Lower is Better)')
    axs[0, 0].set_xlabel('Model Types')
    axs[0, 0].set_ylabel('Root Mean Square Error')
    axs[0, 0].tick_params(axis='x', rotation=45)
    
    # MAE plot
    axs[0, 1].bar(results_df['Model'], results_df['MAE'])
    axs[0, 1].set_title('MAE (Lower is Better)')
    axs[0, 1].set_xlabel('Model Types')
    axs[0, 1].set_ylabel('Mean Absolute Error')
    axs[0, 1].tick_params(axis='x', rotation=45)
    
    # R² plot
    axs[1, 0].bar(results_df['Model'], results_df['R²'])
    axs[1, 0].set_title('R² (Higher is Better)')
    axs[1, 0].set_xlabel('Model Types')
    axs[1, 0].set_ylabel('R-squared Score')
    axs[1, 0].tick_params(axis='x', rotation=45)
    
    # Training time plot
    axs[1, 1].bar(training_times_df['Model'], training_times_df['Training Time (s)'])
    axs[1, 1].set_title('Training Time (Lower is Better)')
    axs[1, 1].set_xlabel('Model Types')
    axs[1, 1].set_ylabel('Training Time (Seconds)')
    axs[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('all_models_comparison.png')
    print(f"\nResults visualization saved to 'all_models_comparison.png'")
    
    # Export results to CSV
    results_df.to_csv('model_comparison_results.csv', index=False)
    training_times_df.to_csv('model_training_times.csv', index=False)
    print("Results exported to CSV files: 'model_comparison_results.csv' and 'model_training_times.csv'")

if __name__ == "__main__":
    main() 