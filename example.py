import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import numpy as np
from model_comparison import ModelComparison

# Generate synthetic time series data
def generate_data(n_samples, sequence_length, n_features):
    X = np.random.randn(n_samples, sequence_length, n_features)
    y = np.sum(X, axis=2).mean(axis=1, keepdims=True)
    return X, y

# Generate data
sequence_length = 10
n_features = 5
n_outputs = 1
n_samples = 1000

X, y = generate_data(n_samples, sequence_length, n_features)

# Split data
train_size = int(0.8 * n_samples)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Initialize and run comparison
model_comparison = ModelComparison(sequence_length, n_features, n_outputs)
results = model_comparison.train_and_evaluate(
    X_train, y_train,
    X_test, y_test,
    epochs=100
)

# Print detailed results
for model_name, metrics in results.items():
    print(f"\n{model_name} Results:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")

# Save visualization
model_comparison.plot_results(results)
import matplotlib.pyplot as plt
plt.savefig('model_comparison_results.png')
print(f"\nResults visualization saved to 'model_comparison_results.png'") 