import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# -----------------------------
# Physical constants (SI units)
# -----------------------------
g = 9.81     # gravity [m/s^2]
L = 0.30     # pendulum length [m]
m = 0.20     # mass [kg]
b = 0.05     # viscous damping [Nms/rad]
J = m * L**2 # inertia [kgm^2]

# -----------------------------
# Control parameters
# -----------------------------
gamma = 0.15      # feedback gain on e_k 
epsilon_v = 0.8   # shrink v each step
alpha_eta = 0.01  # learning rate for v update

# Optional: actuator saturation (uncomment to enable)
U_MAX = 1.0      # [PWM]
USE_SAT = True

# -----------------------------
# Simulation setup
# -----------------------------
t_max = 10.0
dt = 0.001
time = np.arange(0.0, t_max, dt)
N = len(time)

# Reference
theta_ref = np.pi

# System variables
theta = np.pi/3     # initial angle [rad]
omega = 0.0         # initial angular rate [rad/s]
v = 0.0             # adaptive term
u = 0.0             # control torque

theta_log = np.zeros(N)
omega_log = np.zeros(N)
u_log     = np.zeros(N)
v_log     = np.zeros(N)
e_log     = np.zeros(N)

# -----------------------------
# Simulation loop
# -----------------------------
for k in range(N):
    # 1) measure current error e_k
    e_k = theta - theta_ref

    # 2) control u_k = -gamma * e_k + v_k
    u = -gamma * e_k + v
    if USE_SAT:
        u = np.clip(u, -U_MAX, U_MAX)

    # 3) integrate plant with u_k  ->  (theta_{k+1}, omega_{k+1})
    omega = omega + (1.0/J) * (-m*g*L*np.sin(theta) - b*omega + 5*u) * dt
    theta = theta + omega * dt

    # 4) compute e_{k+1}
    e_next = theta - theta_ref

    # 5) adaptation: v_{k+1} = (1 - epsilon_v) v_k - alpha_eta * e_{k+1}
    v = (1.0 - epsilon_v) * v - alpha_eta * e_next

    # log
    theta_log[k] = theta
    omega_log[k] = omega
    u_log[k]     = u
    v_log[k]     = v
    e_log[k]     = e_k

# -----------------------------
# Animation of the pendulum
# -----------------------------
x = L * np.sin(theta_log)
y = -L * np.cos(theta_log)

fig, ax = plt.subplots()

def update(i):
    ax.clear()
    ax.set_xlim(-L-0.1, L+0.1)
    ax.set_ylim(-L-0.1, L+0.1)
    ax.set_aspect('equal')
    ax.plot([0, x[i]], [0, y[i]], 'k-', lw=2)
    ax.plot(x[i], y[i], 'ro', ms=8)
    ax.plot(0, 0, 'bo')  # pivot
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    title_text = (
        f"Pendulum:   t = {time[i]:.2f} s  |  "
        f"theta = {np.degrees(theta_log[i]):.1f}°"
    )
    ax.set_title(title_text, fontsize=12)

skip = 100  # show every skip samples for speed (adjust as you like)
ani = FuncAnimation(fig, update, frames=range(0, N, skip), interval=dt*1000, repeat=False)
plt.show()