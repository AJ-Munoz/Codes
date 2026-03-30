import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import skfuzzy as fuzz

# -----------------------------
# Physical parameters
# -----------------------------
g = 9.81
L = 0.30
m = 0.20
J = m * L**2

# -----------------------------
# Simulation parameters
# -----------------------------
dt = 0.001
t_max = 10.0
time = np.arange(0, t_max, dt)
N = len(time)

# -----------------------------
# Initial conditions
# -----------------------------
theta_real  = 3 * np.pi / 4
omega_real  = 0.0

theta_fuzzy = 3 * np.pi / 4
omega_fuzzy = 0.0

# -----------------------------
# Fuzzy Approximation of sin(theta)
# -----------------------------
n_rules = 64 # Number of fuzzy rules (membership functions)
theta_univ = np.linspace(0, 2*np.pi, 4000) # High-resolution universe for smooth membership functions

centers = np.linspace(0, 2*np.pi, n_rules) # Evenly spaced centers around the circle
sigma = (2*np.pi / n_rules) * 0.7 # 70% of the distance between centers for good coverage

# Taylor Series Coefficients at The Centers
a = np.cos(centers) # Slope (derivative) at the centers
b = np.sin(centers) - centers * np.cos(centers) # Intercept at the centers

# Gaussian Membership Functions
mfs = [fuzz.gaussmf(theta_univ, c, sigma) for c in centers]

def fuzzy_sin(theta):
    """First-order Taylor Takagi–Sugeno approximation of sin(theta)."""
    mu = np.array([
        fuzz.interp_membership(theta_univ, mfs[i], theta)
        for i in range(n_rules)
    ])

    if mu.sum() == 0.0:
        return np.sin(theta)  

    mu /= mu.sum()
    return np.sum(mu * (a * theta + b))

# -----------------------------
# Logs
# -----------------------------
theta_real_log  = np.zeros(N)
theta_fuzzy_log = np.zeros(N)

# -----------------------------
# Simulation loop (open-loop)
# -----------------------------
for k in range(N):

    # Real pendulum
    omega_real += dt * (-m * g * L * np.sin(theta_real)) / J
    theta_real += dt * omega_real

    # Fuzzy pendulum
    omega_fuzzy += dt * (-m * g * L * fuzzy_sin(theta_fuzzy)) / J
    theta_fuzzy += dt * omega_fuzzy

    theta_real_log[k]  = theta_real
    theta_fuzzy_log[k] = theta_fuzzy

# -----------------------------
# Animation
# -----------------------------
x_real  = L * np.sin(theta_real_log)
y_real  = -L * np.cos(theta_real_log)

x_fuzzy = L * np.sin(theta_fuzzy_log)
y_fuzzy = -L * np.cos(theta_fuzzy_log)

fig, ax = plt.subplots()

def update(i):
    ax.clear()
    ax.set_xlim(-L-0.1, L+0.1)
    ax.set_ylim(-L-0.1, L+0.1)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    ax.plot([0, x_real[i]], [0, y_real[i]], 'k-', lw=2, label='Real')
    ax.plot(x_real[i], y_real[i], 'ko')

    ax.plot([0, x_fuzzy[i]], [0, y_fuzzy[i]], 'r--', lw=2, label='Fuzzy (TS–Taylor)')
    ax.plot(x_fuzzy[i], y_fuzzy[i], 'ro')

    ax.legend(loc='upper right')
    ax.set_title(f"Real vs Fuzzy Pendulum   t = {time[i]:.2f} s")

skip = 100
ani = FuncAnimation(fig, update,
                    frames=range(0, N, skip),
                    interval=dt * 1000,
                    repeat=False)

plt.show()