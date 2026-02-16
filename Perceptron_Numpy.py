import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iterations=100):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def predict(self, X):
            # Calculate the weighted sum
            linear_output = np.dot(X, self.weights) + self.bias
            # Apply step function (binary classification)
            # Returns 1 if linear_output >= 0, otherwise -1
            return np.where(linear_output >= 0.0, 1, 0)
            
    def fit(self, X, y):
        # Initialize weights and bias
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Perceptron learning loop
        for _ in range(self.n_iterations):
            for idx, x_i in enumerate(X):
                # Calculate the predicted value (1 or -1)
                y_pred = self.predict(x_i)
                
                # Check for misclassification and update weights
                if y_pred != y[idx]:
                    update = self.learning_rate * (y[idx] - y_pred)
                    self.weights += update * x_i
                    self.bias += update

    

# --- Example Usage (AND gate logic) ---
# Inputs
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# Labels (for AND: [0, 0, 0, 1]. Map 0 to -1 for this implementation)
y = np.array([0, 0, 0, 1])

# Create and train the perceptron
perceptron = Perceptron(learning_rate=0.1, n_iterations=10)
perceptron.fit(X, y)

# Test the trained model
print("AND Gate Predictions:")
for inputs in X:
    prediction = perceptron.predict(inputs)
    print(f"{inputs} -> {prediction}")
