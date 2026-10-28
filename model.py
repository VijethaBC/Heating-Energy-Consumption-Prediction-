import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, Flatten

# Load dataset (update file paths if needed)
df = pd.read_csv('heating_energy_data.csv')  # Replace with actual file

# Select features and target
target = 'heating_energy'
features = ['timestamp', 'indoor_temp_exterior', 'indoor_temp_interior', 'heating_setpoint', 'occupancy_count', 
            'co2_concentration', 'heat_pump_water_temp', 'fan_speed', 'supply_air_temp', 'supply_air_flow_rate',
            'return_air_temp', 'heating_valve_position', 'outdoor_temp', 'outdoor_humidity', 'solar_irradiation', 'wind_speed']

df = df[features + [target]].dropna()

# Convert timestamp to numerical features
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['month'] = df['timestamp'].dt.month
df.drop(columns=['timestamp'], inplace=True)

# Train-test split
X = df.drop(columns=[target])
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)
rf_preds = rf_model.predict(X_test_scaled)

# Train SVR
svr_model = SVR()
svr_model.fit(X_train_scaled, y_train)
svr_preds = svr_model.predict(X_test_scaled)

# Reshape for LSTM/CNN-LSTM
X_train_seq = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
X_test_seq = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))

# Train LSTM
lstm_model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train_seq.shape[1], 1)),
    LSTM(50, return_sequences=False),
    Dense(25, activation='relu'),
    Dense(1)
])
lstm_model.compile(optimizer='adam', loss='mse')
lstm_model.fit(X_train_seq, y_train, epochs=10, batch_size=32, validation_data=(X_test_seq, y_test))
lstm_preds = lstm_model.predict(X_test_seq).flatten()

# Train CNN/LSTM
cnn_lstm_model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train_seq.shape[1], 1)),
    LSTM(50, return_sequences=True),
    LSTM(50, return_sequences=False),
    Dense(25, activation='relu'),
    Dense(1)
])
cnn_lstm_model.compile(optimizer='adam', loss='mse')
cnn_lstm_model.fit(X_train_seq, y_train, epochs=10, batch_size=32, validation_data=(X_test_seq, y_test))
cnn_lstm_preds = cnn_lstm_model.predict(X_test_seq).flatten()

# Evaluate models
def evaluate_model(y_true, y_pred, model_name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    print(f'{model_name} -> RMSE: {rmse:.4f}, MAE: {mae:.4f}')

evaluate_model(y_test, rf_preds, 'Random Forest')
evaluate_model(y_test, svr_preds, 'SVR')
evaluate_model(y_test, lstm_preds, 'LSTM')
evaluate_model(y_test, cnn_lstm_preds, 'CNN/LSTM')
