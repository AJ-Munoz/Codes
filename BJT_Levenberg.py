import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# 1. Input Data
# Input Data: BJT I-V measurements with added noise for realism
vbe_data = np.array([0.4, 0.433, 0.467, 0.5, 0.533, 0.567, 0.6, 0.633, 0.667, 0.7])
ic_data = np.array([2.33e-6, 7.80e-6, 2.69e-5, 9.27e-5, 3.04e-4, 1.03e-3, 3.62e-3, 1.21e-2, 3.99e-2, 1.38e-1])

# Add reproducible small noise to Vbe and Ic
rng = np.random.default_rng(seed=42)
vbe_data = vbe_data + rng.uniform(-2e-3, 2e-3, size=vbe_data.shape)
ic_data = ic_data + rng.uniform(-2e-3, 2e-3, size=ic_data.shape)

# 2. BJT Shockley Model (Thermal voltage Vt ~ 0.026V)
def bjt_model(params, v_be):
    is_val, n = params
    return is_val * (np.exp(v_be / (n * 0.026)) - 1)

# 3. Residual Function
def residuals(params, v_be, ic_data):
    return ic_data - bjt_model(params, v_be)

# 4. Levenberg-Marquardt
initial_guess = [1e-12, 1.0]
res = least_squares(residuals, initial_guess, args=(vbe_data, ic_data), method='lm')

print(f"Fitted Is: {res.x[0]:.4e} A")
print(f"Fitted n: {res.x[1]:.4f}")
print(f"Cost: {res.cost:.4e}")

# --- VISUALIZATION ---
plt.figure()
vbe_fit = np.linspace(vbe_data.min(), vbe_data.max(), 100)
ic_fit = bjt_model(res.x, vbe_fit)
plt.plot(vbe_fit, ic_fit, 'b-', label='LM Fit')
plt.scatter(vbe_data, ic_data, color='red', label='Data')
plt.title('BJT I-V Characteristic')
plt.xlabel('Vbe (V)')
plt.ylabel('Ic (A)')
plt.legend() 
plt.grid(True)
plt.tight_layout()
plt.show()