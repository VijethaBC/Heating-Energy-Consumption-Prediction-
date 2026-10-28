import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, bidirectional=False):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, 
                         bidirectional=bidirectional)
        mult = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_size * mult, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), 
                        x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out

class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(TCNBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                            padding=(kernel_size-1) * dilation,
                            dilation=dilation)
        self.norm = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.relu(out)
        out = self.dropout(out)
        return out

class TCNModel(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size):
        super(TCNModel, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers.append(
                TCNBlock(in_channels, out_channels, kernel_size, dilation)
            )
        
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], output_size)
        
    def forward(self, x):
        # Input shape: [batch, seq_len, features]
        # Reshape for Conv1d: [batch, features, seq_len]
        x = x.transpose(1, 2)
        out = self.tcn(x)
        out = out[:, :, -1]  # Take last output
        out = self.fc(out)
        return out

class ModelComparison:
    def __init__(self, sequence_length, n_features, n_outputs):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def create_bigru_model(self):
        return GRUModel(self.n_features, 64, 2, self.n_outputs, bidirectional=True).to(self.device)
    
    def create_gru_model(self):
        return GRUModel(self.n_features, 64, 2, self.n_outputs, bidirectional=False).to(self.device)
    
    def create_tcn_model(self):
        num_channels = [64, 32]
        return TCNModel(self.n_features, self.n_outputs, num_channels, kernel_size=3).to(self.device)
    
    def train_model(self, model, X_train, y_train, epochs=100):
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters())
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.FloatTensor(y_train).to(self.device)
        
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
        
        return model
    
    def predict(self, model, X):
        model.eval()
        with torch.no_grad():
            X = torch.FloatTensor(X).to(self.device)
            outputs = model(X)
            return outputs.cpu().numpy()
    
    def train_and_evaluate(self, X_train, y_train, X_test, y_test, epochs=100):
        results = {}
        
        # BiGRU
        bigru = self.create_bigru_model()
        bigru = self.train_model(bigru, X_train, y_train, epochs)
        bigru_pred = self.predict(bigru, X_test)
        results['BiGRU'] = self.calculate_metrics(y_test, bigru_pred)
        
        # GRU
        gru = self.create_gru_model()
        gru = self.train_model(gru, X_train, y_train, epochs)
        gru_pred = self.predict(gru, X_test)
        results['GRU'] = self.calculate_metrics(y_test, gru_pred)
        
        # TCN
        tcn = self.create_tcn_model()
        tcn = self.train_model(tcn, X_train, y_train, epochs)
        tcn_pred = self.predict(tcn, X_test)
        results['TCN'] = self.calculate_metrics(y_test, tcn_pred)
        
        return results
    
    def calculate_metrics(self, y_true, y_pred):
        return {
            'MSE': mean_squared_error(y_true, y_pred),
            'MAE': mean_absolute_error(y_true, y_pred),
            'R2': r2_score(y_true, y_pred)
        }
    
    def plot_results(self, results):
        metrics = ['MSE', 'MAE', 'R2']
        models = list(results.keys())
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, metric in enumerate(metrics):
            values = [results[model][metric] for model in models]
            axes[i].bar(models, values)
            axes[i].set_title(metric)
            axes[i].set_xlabel('Models')
            axes[i].set_ylabel('Score')
            axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('model_comparison_results.png')
        plt.show()

if __name__ == "__main__":
    # Generate sample data
    sequence_length = 10
    n_features = 5
    n_outputs = 1
    n_samples = 1000
    
    # Create random sample data
    X = np.random.randn(n_samples, sequence_length, n_features)
    y = np.random.randn(n_samples, n_outputs)
    
    # Split data
    train_size = int(0.8 * n_samples)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Initialize and run comparison
    model_comparison = ModelComparison(sequence_length, n_features, n_outputs)
    results = model_comparison.train_and_evaluate(X_train, y_train, X_test, y_test)
    
    # Print results
    for model_name, metrics in results.items():
        print(f"\n{model_name} Results:")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")
    
    # Plot results
    model_comparison.plot_results(results) 