import numpy as np
from scipy.linalg import pinv
import matplotlib.pyplot as plt


np.random.seed(7)

# Coordinate and true weights
n = 200; x = np.linspace(0, 1, 200)
w_true = np.array([1.5, -1.5, 0.5])  # True weights

# Nonlinear features f_i(x)
f1 = np.sin(np.pi * x)
f2 = np.sin(2 * np.pi * x)
f3 = np.cos(8 * np.pi * x)

# Design matrix F with columns f1, f2, f3
F = np.column_stack([f1, f2, f3])

# Clean signal and noisy measurement
y_clean = F @ w_true
noise = 0.5 * np.random.randn(n)
y = y_clean + noise

# Pseudoinverse solution w
w_hat = pinv(F) @ y      # shape (3,)
y_hat = F @ w_hat        # denoised projection onto span{f1,f2,f3}

print("Estimated weights:", w_hat)

# Visualize the results
plt.figure(figsize=(10, 6))
plt.plot(x, y,        alpha=0.5, lw=2.0, label="noisy")
plt.plot(x, y_clean,  alpha=0.9, lw=2.0, label="clean (true)")
plt.plot(x, y_hat,    alpha=0.9, lw=2.0, label="denoised (projection)")

plt.xlabel("span x"); plt.ylabel("response")
plt.legend(); plt.tight_layout(); plt.show()