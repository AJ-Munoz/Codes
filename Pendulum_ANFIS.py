import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import skfuzzy as fuzz

# =========================================================
# Configuration
# =========================================================
dt = 0.001
TARGET_ANGLE = np.pi  

state_adaptive = np.array([0.0, 0.0])
state_fixed = np.array([0.0, 0.0])

# Fuzzy Parameters
CENTERS = np.array([-1.0, 0.0, 1.0])
SIGMA = 0.5
NUM_RULES = 27
weights = np.zeros(NUM_RULES)
GAMMA = 150.0  # Learning rate

# Memory
error_int_a = 0.0
prev_err_a = 0.0
prev_err_f = 0.0

def fuzzy_basis_sk(error, error_dot, error_int):
    # 1. Calculate membership values for each input and each center
    m_e = [fuzz.gaussmf(error, c, SIGMA) for c in CENTERS]
    m_de = [fuzz.gaussmf(error_dot, c, SIGMA) for c in CENTERS]
    m_ie = [fuzz.gaussmf(error_int, c, SIGMA) for c in CENTERS]

    # 2. Combine into rules (Product T-Norm)
    phi = np.array([
        m_e[i] * m_de[j] * m_ie[k]
        for i in range(3) for j in range(3) for k in range(3)
    ])

    total = np.sum(phi)
    return phi / total if total > 1e-9 else phi

def pendulum_step(state, u):
    theta, theta_dot = state
    accel = u - 10.0 * np.sin(theta) - 0.5 * theta_dot
    theta_dot += accel * dt
    theta += theta_dot * dt
    return np.array([theta, theta_dot])

# =========================================================
# Animation
# =========================================================
fig, ax = plt.subplots(figsize=(5, 5))
ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5); ax.set_aspect('equal')
line_a, = ax.plot([], [], 'ro-', lw=2, label='ANFIS PID')
line_f, = ax.plot([], [], 'bo-', lw=2, alpha=0.3, label='Fuzzy PD')
ax.legend()

def update(frame):
    global state_adaptive, state_fixed, weights
    global error_int_a, prev_err_a, prev_err_f

    # --- Poorly Tuned Gains ---
    kp, kd = 2.0, 5.0

    # --- Adaptive Logic ---
    err_a = TARGET_ANGLE - state_adaptive[0]
    err_dot_a = (err_a - prev_err_a) / dt
    error_int_a += err_a * dt
    
    phi = fuzzy_basis_sk(err_a, err_dot_a, error_int_a * 0.1)
    
    # SLIDING SURFACE: Alleviates the overshoot
    S = err_dot_a + 2.0 * err_a 
    
    # Adaptive Law: Learn based on S, not just error
    weights += (GAMMA * S * phi - 0.1 * weights) * dt
    u_a = np.dot(weights, phi) + (kp * err_a + kd * err_dot_a)

    # --- Fixed PD Logic ---
    err_f = TARGET_ANGLE - state_fixed[0]
    err_dot_f = (err_f - prev_err_f) / dt
    u_f = kp * err_f + kd * err_dot_f

    # --- Physics & Memory Update ---
    state_adaptive = pendulum_step(state_adaptive, u_a)
    state_fixed = pendulum_step(state_fixed, u_f)
    prev_err_a, prev_err_f = err_a, err_f

    # --- Drawing both Pendulums ---
    line_a.set_data([0, np.sin(state_adaptive[0])], [0, -np.cos(state_adaptive[0])])
    line_f.set_data([0, np.sin(state_fixed[0])], [0, -np.cos(state_fixed[0])])

    return line_a, line_f

skip = 10
ani = FuncAnimation(fig, update, frames=range(0, 2000, skip), interval=dt*1000, blit=True)
plt.grid(True, alpha=0.3)
plt.show()
