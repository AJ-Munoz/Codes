import numpy as np
import matplotlib.pyplot as plt

# 1. Create a signal with a sharper jump
x = np.linspace(0, 10, 100)
f = np.where(x < 5, np.sin(x), np.cos(x))

# 2. Difference filter
h = np.array([-1, 0, 1])

# 3. Apply convolution with 'same' mode
edge = np.convolve(f, h, mode='same')

# 4. Visualization
plt.figure(figsize=(10, 5))
plt.plot(x, f, label="Original Signal (f)", linewidth=2.5, alpha=0.8)
plt.plot(x, edge, label="Edge Response (h * f)", linewidth=2, color="crimson")
plt.fill_between(x, edge, color="crimson", alpha=0.2)

# Styling
plt.title("1D Edge Detection: Finding the 'Jump'", fontsize=14, pad=15)
plt.axvline(5, color='gray', linestyle='--', alpha=0.5, label="True Edge Location")
plt.xlabel("x")
plt.ylabel("Amplitude")
plt.legend(frameon=True, loc='upper left')
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()

plt.show()