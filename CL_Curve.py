import numpy as np
from scipy.linalg import pinv
import matplotlib.pyplot as plt

np.random.seed(7)

# Synthetic wind-tunnel data
N = 250; 
angle = np.linspace(-15, 25, N) * np.pi/180.0 
q = 0.2 + 0.05*np.sin(4*np.pi*np.linspace(0,1,N)) 

# True parameters (unknown in practice)
w_true = np.array([0.1, 6.2, -35.47, 0.023])  

# Design matrix: [1, angle, angle^3, q]
F = np.column_stack([np.ones(N), angle, angle**3, q])

# Generate C_L with noise
CL_clean = F @ w_true
CL = CL_clean + 0.1*np.random.randn(N)

# Pseudoinverse solution w
w_hat = pinv(F) @ CL      # shape (4,)
CL_hat = F @ w_hat        # denoised projection onto span{f1,f2,f3}

print("Estimated weights:", w_hat)

# Visualize the results
plt.figure(figsize=(10, 6))
plt.title("Pseudoinverse-based Denoising", fontsize=16)
plt.plot(angle, CL,        alpha=0.5, lw=2.0, label="noisy")
plt.plot(angle, CL_clean,  alpha=0.9, lw=2.0, label="clean (unknown)")
plt.plot(angle, CL_hat,    alpha=0.9, lw=2.0, label="denoised")

plt.xlabel("angle", fontsize=16) 
plt.ylabel("Lift Coefficient (C_L)", fontsize=16)
plt.legend()
plt.tight_layout()
plt.grid()
plt.show()