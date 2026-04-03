import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import skfuzzy as fuzz

# --- Configuration ---
dt, steps = 0.01, 2000
time_steps = np.arange(0, steps * dt, dt)
target = np.pi  # Target angle (0 is vertical down, let's aim for 0 for stability first)
s_adaptive = np.array([0.78, 0.0]) # Start at ~45 degrees
s_fixed = np.array([0.78, 0.0])

# --- Fuzzy Setup ---
# Narrower centers and sigma for better sensitivity
centers = np.array([-0.5, 0.0, 0.5]) 
sigma = 0.3
num_rules = 27 
beta = np.zeros(num_rules) # Renamed from theta
gamma = 100.0  # Increased learning rate for visible adaptation

# --- Simulation Variables ---
e_int_a = 0.0
e_prev_a = 0.78

def get_phi(e, de, ie):
    # Gaussian Membership
    m_e = [fuzz.gaussmf(e, c, sigma) for c in centers]
    m_de = [fuzz.gaussmf(de, c, sigma) for c in centers]
    m_ie = [fuzz.gaussmf(ie, c, sigma) for c in centers]
    
    # Rule Strengths (Product T-norm)
    w = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                w.append(m_e[i] * m_de[j] * m_ie[k])
    
    phi = np.array(w)
    total_w = np.sum(phi)
    return phi / total_w if total_w > 1e-9 else phi

def update(frame):
    global s_adaptive, s_fixed, e_int_a, e_prev_a, beta
    
    # 1. ADAPTIVE CONTROLLER
    err_a = target - s_adaptive[0]
    de_a = (err_a - e_prev_a) / dt
    e_int_a += err_a * dt
    
    phi = get_phi(err_a, de_a, e_int_a)
    u_a = np.dot(beta, phi) + 15 * err_a + 5 * (err_a - (target - s_adaptive[0]))/dt
    
    # Adaptation Law (Lyapunov-based)
    beta += gamma * (err_a * phi) * dt
    
    # 2. FIXED PID (for comparison)
    err_f = target - s_fixed[0]
    u_f = 15 * err_f + 5 * (err_f - (target - s_fixed[0]))/dt # Simple P+D
    
    # 3. PHYSICS (Inverted Pendulum)
    def step(s, u):
        # Dampened pendulum: accel = u - gravity - friction
        accel = u - 10.0 * np.sin(s[0]) - 1.0 * s[1]
        s[1] += accel * dt
        s[0] += s[1] * dt
        return s

    s_adaptive = step(s_adaptive, u_a)
    s_fixed = step(s_fixed, u_f)
    
    # Update Visuals
    line_a.set_data([0, np.sin(s_adaptive[0])], [0, -np.cos(s_adaptive[0])])
    line_f.set_data([0, np.sin(s_fixed[0])], [0, -np.cos(s_fixed[0])])
    e_prev_a = err_a
    return line_a, line_f

# --- Animation ---
fig, ax = plt.subplots(figsize=(6,6))
ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
line_a, = ax.plot([], [], 'ro-', lw=3, label='Adaptive (Beta)')
line_f, = ax.plot([], [], 'bo-', lw=2, alpha=0.3, label='Fixed PID')
ax.legend()
ax.grid(True, alpha=0.3)

ani = FuncAnimation(fig, update, frames=steps, interval=10, blit=True)
plt.show()
