import numpy as np
from scipy.linalg import pinv
import matplotlib.pyplot as plt

# Synthetic data: y = a*t + b
np.random.seed(42)
t = np.linspace(0, 10, 40)
y = 3.2 * t + 1.5 + 1.5 * np.random.randn(len(t))

# Build design matrix F = [t, 1]
F = np.column_stack([np.ones_like(t), t])

# Pseudoinverse solution
w = pinv(F) @ y
w0_hat, w1_hat = w

print("Estimated slope:", w1_hat)
print("Estimated intercept:", w0_hat)

# Prediction
y_hat = F @ w

# Visualization
neonblue = (0/255, 150/255, 220/255) # Blue for the fit line
neonpink = (255/255, 105/255, 180/255) # Neon pink for data points
plt.scatter(t, y, s=30, color=neonpink, alpha=0.9, label="noisy data")
plt.plot(t, y_hat, color=neonblue, lw=3, label="least-squares fit")
plt.xlabel("t", fontsize=16); plt.ylabel("y", fontsize=16)
plt.legend()
plt.tight_layout()
plt.grid()
plt.show()