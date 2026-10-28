import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional, List
import requests
from datetime import datetime
import zipfile
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.scaler = StandardScaler()
        
    def download_file(self, url: str, local_path: str) -> str:
        """Download a file from URL"""
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024
            
            logger.info(f"Downloading {url} to {local_path}")
            with open(local_path, 'wb') as f:
                for data in response.iter_content(block_size):
                    f.write(data)
                    
            logger.info(f"Download completed: {local_path}")
            return local_path
            
        except Exception as e:
            logger.error(f"Error downloading {url}: {str(e)}")
            raise
    
    def download_dataset(self, dataset_name: str = "uci_energy") -> List[str]:
        """
        Download dataset from various sources
        Currently supports:
        - uci_energy
        Returns a list of paths to the downloaded files
        """
        dataset_paths = []
        
        if dataset_name == "uci_energy":
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00374/energydata_complete.csv"
            local_path = os.path.join(self.data_dir, "energydata_complete.csv")
            if not os.path.exists(local_path):
                self.download_file(url, local_path)
            dataset_paths.append(local_path)
        else:
            raise ValueError(f"Dataset {dataset_name} not supported")
            
        return dataset_paths

    def merge_datasets(self, file_paths: List[str]) -> pd.DataFrame:
        """Merge multiple datasets into one"""
        dfs = []
        for path in file_paths:
            df = pd.read_csv(path)
            dfs.append(df)
            
        return pd.concat(dfs, axis=0, ignore_index=True)

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the data by handling missing values and feature engineering"""
        # Handle missing values
        df = df.ffill()  # Forward fill
        df = df.bfill()  # Backward fill for any remaining NaNs
        
        # Convert date column to timestamp if it exists
        if 'date' in df.columns:
            df['timestamp'] = pd.to_datetime(df['date'])
            df.drop(columns=['date'], inplace=True)
        
        # Add time-based features if timestamp exists
        if 'timestamp' in df.columns:
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['month'] = df['timestamp'].dt.month
            df['is_weekend'] = df['timestamp'].dt.dayofweek.isin([5, 6]).astype(int)
            df['season'] = (df['timestamp'].dt.month % 12 + 3) // 3
            df.drop(columns=['timestamp'], inplace=True)
        
        # Drop any random variables (rv1, rv2) as they are not meaningful features
        if 'rv1' in df.columns:
            df.drop(columns=['rv1'], inplace=True)
        if 'rv2' in df.columns:
            df.drop(columns=['rv2'], inplace=True)
            
        return df

    def prepare_sequences(self, data: np.ndarray, target: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for LSTM/CNN models"""
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:(i + sequence_length)])
            y.append(target[i + sequence_length])
        return np.array(X), np.array(y)

    def load_and_preprocess(self, dataset_names: List[str], target_column: str, 
                          sequence_length: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Load, preprocess and prepare the data"""
        all_file_paths = []
        for dataset_name in dataset_names:
            file_paths = self.download_dataset(dataset_name)
            all_file_paths.extend(file_paths)
            
        # Merge all datasets
        df = self.merge_datasets(all_file_paths)
        
        # Preprocess
        df = self.preprocess_data(df)
        
        # Ensure target column exists
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in the dataset")
        
        # Split features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        if sequence_length:
            # Prepare sequences for LSTM/CNN
            X_seq, y_seq = self.prepare_sequences(X_scaled, y.values, sequence_length)
            return X_seq, y_seq
        
        return X_scaled, y.values 