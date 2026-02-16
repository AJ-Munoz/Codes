import numpy as np

# ---------------------------
# Utilities
# ---------------------------
def sigmoid(x):
    # Numerically safer sigmoid via clipping to avoid overflow in exp
    x = np.clip(x, -40, 40)
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_derivative(x):
    # Derivative with respect to pre-activation input x
    s = sigmoid(x)
    return s * (1.0 - s)

# ---------------------------
# 1) Data: XOR
# ---------------------------
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]], dtype=float)

y = np.array([[0],
              [1],
              [1],
              [0]], dtype=float)

# ---------------------------
# 2) Parameters: weights & biases
# Architecture: 2 -> 2 -> 1, sigmoid activations
# ---------------------------
np.random.seed(42)
w_hidden = np.random.uniform(low=-1.0, high=1.0, size=(2, 2))  # (in=2, hidden=2)
b_hidden = np.random.uniform(low=-1.0, high=1.0, size=(1, 2))  # (2, hidden)
w_output = np.random.uniform(low=-1.0, high=1.0, size=(2, 1))  # (hidden=2, out=1)
b_output = np.random.uniform(low=-1.0, high=1.0, size=(1, 1))  # (1, out)

lr = 0.5
epochs = 1000

# ---------------------------
# 3) Training Loop (batch size = 4 samples)
# ---------------------------
for epoch in range(epochs):
    # --- Forward ---
    z_hidden = X @ w_hidden + b_hidden       # (4,2)
    v_hidden = sigmoid(z_hidden)             # (4,2)

    z_output = v_hidden @ w_output + b_output  # (4,1)
    y_hat = sigmoid(z_output)                  # (4,1)

    # --- Backward ---
    # Using error = (target - prediction) so that adding the update moves along -grad
    error = y - y_hat                                    # (4,1)
    d_output = error * sigmoid_derivative(z_output)      # (4,1)

    error_hidden = d_output @ w_output.T                    # (4,2)
    d_hidden = error_hidden * sigmoid_derivative(z_hidden)  # (4,2)

    # --- Update ---
    w_output += (v_hidden.T @ d_output) * lr                  # (2,1)
    b_output += np.sum(d_output, axis=0, keepdims=True) * lr  # (1,1)

    w_hidden += (X.T @ d_hidden) * lr                         # (2,2)
    b_hidden += np.sum(d_hidden, axis=0, keepdims=True) * lr  # (1,2)

    # Optional: print progress occasionally
    # if (epoch + 1) % 200 == 0:
    #     mse = np.mean((y - y_hat) ** 2)
    #     print(f"Epoch {epoch+1:4d} | MSE: {mse:.6f}")

# ---------------------------
# 4) Results
# ---------------------------
print("Final Predictions (raw probabilities):")
print(np.round(y_hat, 6))
print("\nRounded Predictions:")
print(np.round(y_hat))