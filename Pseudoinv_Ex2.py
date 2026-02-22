import numpy as np
from scipy.linalg import pinv
import matplotlib.pyplot as plt

np.random.seed(1)

# Modal shape: e.g., dominant bending mode
n = 200
x = np.linspace(0, 1, n)
mode_shape = np.sin(3 * np.pi * x)  # known aerodynamic mode

# True modal amplitude
A_true = 2.5

# Clean signal
clean = A_true * mode_shape

# Add aerodynamic turbulence noise
noise = 0.6 * np.random.randn(n)
y = clean + noise

# Design matrix: one basis vector (the mode shape)
F = mode_shape[:, None]

# Pseudoinverse estimate of modal amplitude
A_hat = float(pinv(F) @ y)

denoised = A_hat * mode_shape

print("True amplitude:", A_true)
print("Estimated amplitude:", A_hat)

# (Optional plotting)
plt.plot(x, y, label="noisy")
plt.plot(x, denoised, label="denoised")
plt.legend()
plt.show()