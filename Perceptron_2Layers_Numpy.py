import numpy as np

# Sigmoid and its derivative
def sigmoid(x): return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x): s = sigmoid(x); return s * (1 - s)

# 1. Setup Data (XOR)
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])

# 2. Initialize Weights & Biases
np.random.seed(42)
w_hidden = np.random.uniform(size=(2, 2))
b_hidden = np.random.uniform(size=(1, 2))
w_output = np.random.uniform(size=(2, 1))
b_output = np.random.uniform(size=(1, 1))

lr = 0.5

# 3. Training Loop
for epoch in range(1000): # Increased epochs for better convergence
    # --- Forward Pass ---
    z_hidden = np.dot(X, w_hidden) + b_hidden
    v_hidden = sigmoid(z_hidden)
    
    z_output = np.dot(v_hidden, w_output) + b_output
    y_hat = sigmoid(z_output)

    # --- Backward Pass ---
    # Error at output (Target - Predicted)
    error = y - y_hat
    d_output = error * sigmoid_derivative(z_output)

    # Error at hidden layer
    error_hidden = d_output.dot(w_output.T)
    d_hidden = error_hidden * sigmoid_derivative(z_hidden)

    # --- Update Weights & Biases ---
    w_output += v_hidden.T.dot(d_output) * lr
    # Summing across axis 0 ensures we get one bias update per neuron
    b_output += np.sum(d_output, axis=0, keepdims=True) * lr
    
    w_hidden += X.T.dot(d_hidden) * lr
    b_hidden += np.sum(d_hidden, axis=0, keepdims=True) * lr

# 4. Results
print("Final Predictions (Raw):")
print(y_hat)
print("\nRounded Predictions:")
print(np.round(y_hat))
