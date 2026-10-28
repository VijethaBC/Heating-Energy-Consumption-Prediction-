import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from typing import Dict, Any, Tuple

class ModelFactory:
    @staticmethod
    def create_random_forest(params: Dict[str, Any] = None) -> RandomForestRegressor:
        default_params = {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42
        }
        if params:
            default_params.update(params)
        return RandomForestRegressor(**default_params)

    @staticmethod
    def create_svr(params: Dict[str, Any] = None) -> SVR:
        default_params = {
            'kernel': 'rbf',
            'C': 1.0,
            'epsilon': 0.1,
            'gamma': 'scale'
        }
        if params:
            default_params.update(params)
        return SVR(**default_params)

    @staticmethod
    def create_lstm(input_shape: Tuple[int, int], params: Dict[str, Any] = None) -> Sequential:
        default_params = {
            'lstm_units': [50, 25],
            'dense_units': [16],
            'dropout_rate': 0.2,
            'learning_rate': 0.002
        }
        if params:
            default_params.update(params)

        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(default_params['lstm_units'][0], 
                      return_sequences=True,
                      input_shape=input_shape))
        model.add(Dropout(default_params['dropout_rate']))
        
        # Second LSTM layer
        model.add(LSTM(default_params['lstm_units'][1]))
        model.add(Dropout(default_params['dropout_rate']))
        
        # Dense layers
        for units in default_params['dense_units']:
            model.add(Dense(units, activation='relu'))
            model.add(Dropout(default_params['dropout_rate']))
        
        # Output layer
        model.add(Dense(1))
        
        model.compile(
            optimizer=Adam(learning_rate=default_params['learning_rate']),
            loss='mse'
        )
        
        return model

    @staticmethod
    def create_cnn_lstm(input_shape: Tuple[int, int], params: Dict[str, Any] = None) -> Sequential:
        default_params = {
            'conv_filters': [32],
            'conv_kernel_size': 3,
            'lstm_units': [25],
            'dense_units': [16],
            'dropout_rate': 0.2,
            'learning_rate': 0.002
        }
        if params:
            default_params.update(params)

        model = Sequential()
        
        # CNN layers
        model.add(Conv1D(filters=default_params['conv_filters'][0],
                       kernel_size=default_params['conv_kernel_size'],
                       activation='relu',
                       input_shape=input_shape))
        model.add(Dropout(default_params['dropout_rate']))
        
        # LSTM layer
        model.add(LSTM(default_params['lstm_units'][0]))
        model.add(Dropout(default_params['dropout_rate']))
        
        # Dense layers
        for units in default_params['dense_units']:
            model.add(Dense(units, activation='relu'))
            model.add(Dropout(default_params['dropout_rate']))
        
        # Output layer
        model.add(Dense(1))
        
        model.compile(
            optimizer=Adam(learning_rate=default_params['learning_rate']),
            loss='mse'
        )
        
        return model

    @staticmethod
    def get_callbacks():
        return [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]

    @staticmethod
    def tune_random_forest(model: RandomForestRegressor, X_train: np.ndarray, y_train: np.ndarray) -> RandomForestRegressor:
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=3,
            n_jobs=-1,
            verbose=2,
            scoring='neg_mean_squared_error'
        )
        
        grid_search.fit(X_train, y_train)
        return grid_search.best_estimator_

    @staticmethod
    def tune_svr(model: SVR, X_train: np.ndarray, y_train: np.ndarray) -> SVR:
        param_grid = {
            'C': [0.1, 1.0],
            'epsilon': [0.1],
            'gamma': ['scale', 'auto']
        }
        
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=3,
            n_jobs=-1,
            verbose=2,
            scoring='neg_mean_squared_error'
        )
        
        grid_search.fit(X_train, y_train)
        return grid_search.best_estimator_ 