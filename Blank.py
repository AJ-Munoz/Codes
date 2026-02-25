import numpy as np
import matplotlib.pyplot as plt

# 1. Sigmoid Function: 1 / (1 + e^-x)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 2. Hyperbolic Tangent: (e^x - e^-x) / (e^x + e^-x)
# Note: This is also available as np.tanh(x)
def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

# 3. Rectified Linear Unit: max(0, x)
def relu(x):
    return np.maximum(0, x)

# Generate data points
x = np.linspace(-5, 5, 100)

# Plotting the functions
plt.figure(figsize=(10, 6))


plt.plot(x, sigmoid(x), label='Sigmoid', color='magenta', linewidth=2)
plt.plot(x, tanh(x), label='Tanh', color='cyan', linewidth=2)
plt.plot(x, relu(x), label='ReLU', color='lime', linewidth=2)

plt.axhline(0, color='white', linewidth=0.5)
plt.axvline(0, color='white', linewidth=0.5)
plt.title('Common Activation Functions', color='white')
plt.legend(fontsize=14)
plt.grid(alpha=0.3)

# Style adjustment for dark background like the image
plt.gca().set_facecolor([10/255, 10/255, 10/255])  # Dark red background
plt.gcf().set_facecolor([10/255, 10/255, 10/255])
plt.tick_params(colors='white')

plt.axis([-5.1, 5.1, -1.5, 3.5])
plt.show()
