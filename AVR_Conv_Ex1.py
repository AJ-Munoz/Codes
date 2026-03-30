import numpy as np
import matplotlib.pyplot as plt

# 1. Generate a noisier signal to better see the "averaging" effect
x = np.linspace(0, 10, 20)
f = np.sin(x) + np.random.normal(-0.5, 0.5, 20)

# 2. Define filter (Window size 3 for more visible smoothing)
window_size = 3
g = np.ones(window_size) / window_size

# 3. Use mode='same' to keep output length equal to input length
# This ensures the indices (x-axis) match perfectly
conv = np.convolve(f, g, mode='same')

# 4. Improved Plotting
plt.figure(figsize=(10, 5))

plt.plot(f, 'o-', alpha=0.4, label='Original (Noisy)', markersize=4)
plt.plot(conv, 'r-', linewidth=2, label=f'Moving Average (n={window_size})')

plt.title("Signal Smoothing via Convolution")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

plt.tight_layout()
plt.show()
