import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def get_W(phi, theta):
    return np.array([
        [1,            0,              -np.sin(theta)],
        [0,  np.cos(phi), np.sin(phi) * np.cos(theta)],
        [0, -np.sin(phi), np.cos(phi) * np.cos(theta)]
    ])

def damped_inverse(W, rho=0.1):
    return np.linalg.inv(W.T @ W + rho**2 * np.eye(3)) @ W.T

def get_R(phi, theta, psi):
    Rx = np.array([[1, 0, 0], [0, np.cos(phi), -np.sin(phi)], [0, np.sin(phi), np.cos(phi)]])
    Ry = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
    Rz = np.array([[np.cos(psi), -np.sin(psi), 0], [np.sin(psi), np.cos(psi), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx

# --- Satellite Geometry ---
def get_cube_faces(center, size):
    s = size / 2
    x, y, z = center
    return [
        [[x-s,y-s,z-s], [x+s,y-s,z-s], [x+s,y+s,z-s], [x-s,y+s,z-s]], # Bottom
        [[x-s,y-s,z+s], [x+s,y-s,z+s], [x+s,y+s,z+s], [x-s,y+s,z+s]], # Top
        [[x-s,y-s,z-s], [x+s,y-s,z-s], [x+s,y-s,z+s], [x-s,y-s,z+s]], # Front
        [[x-s,y+s,z-s], [x+s,y+s,z-s], [x+s,y+s,z+s], [x-s,y+s,z+s]], # Back
        [[x-s,y-s,z-s], [x-s,y+s,z-s], [x-s,y+s,z+s], [x-s,y-s,z+s]], # Left
        [[x+s,y-s,z-s], [x+s,y+s,z-s], [x+s,y+s,z+s], [x+s,y-s,z+s]]  # Right
    ]

# Setup simulation
dt, steps = 0.001, 10000 # 10 seconds at 1ms steps
eta = np.array([0.0, 0.0, 0.0])  # Initial orientation (roll, pitch, yaw)
omega_cmd = np.array([0.2, 0.1, 0.3])
history = []
time = [0]

# Simulation Loop
for i in range(steps):
    t_curr = time[-1]
    W_inv = damped_inverse(get_W(eta[0], eta[1]), 0.001)
    #np.sin(np.pi * t_curr)
    roll =  np.pi       #
    pitch = np.pi       #
    yaw = np.pi         #
    eta_ref = np.array([roll, pitch, yaw])  # Desired orientation 
    omega_cmd = - 1.0 * get_W(eta[0], eta[1]) @ (eta - eta_ref)  # Proportional control to track reference
    eta += (W_inv @ omega_cmd) * dt
    history.append(eta.copy())
    time.append(time[-1] + dt)

# USE THIS FOR ANIMATION (GEMINI-GENERATED) INSTEAD OF STATIC PLOTS
# --- Pre-generate Stars ---
num_stars = 500
star_coords = (np.random.rand(num_stars, 3) - 0.5) * 8 
star_sizes = np.random.rand(num_stars) * 7            

fig = plt.figure(figsize=(10, 8), facecolor='#050505')
ax = fig.add_subplot(111, projection='3d', facecolor='#050505')

def update(i):
    ax.clear()
    # Keep the view fixed so the satellite rotates, not the stars
    ax.set_xlim([-2, 2]); ax.set_ylim([-2, 2]); ax.set_zlim([-2, 2])
    ax.axis('off')

    # 0. Draw the Stars
    ax.scatter(star_coords[:, 0], star_coords[:, 1], star_coords[:, 2], 
               c='white', s=star_sizes, alpha=0.8, edgecolors='none', antialiased=True)

    R = get_R(*history[i])
    
    # 1. Draw Satellite Bus
    bus_faces = np.array(get_cube_faces([0,0,0], 0.6))
    rotated_bus = [ (R @ face.T).T for face in bus_faces ]
    ax.add_collection3d(Poly3DCollection(rotated_bus, facecolors="#76027EF8", 
                                         linewidths=1, edgecolors='#B8860B', alpha=0.9))
    
    # 2. Draw Solar Panels
    panel_width, panel_length = 0.5, 1.2
    for side in [-1, 1]: 
        p_faces = [np.array([
            [0, side*0.3, -panel_width/2], [0, side*(0.3+panel_length), -panel_width/2],
            [0, side*(0.3+panel_length), panel_width/2], [0, side*0.3, panel_width/2]
        ])]
        rot_panel = [ (R @ f.T).T for f in p_faces ]
        ax.add_collection3d(Poly3DCollection(rot_panel, facecolors="#0E1448", 
                                             edgecolors='silver', alpha=1.0))

    title_text = (
        f"Orbital Tracking (Gemini-Generated)\n"
        f"time: {time[i]:.2f}s | "
        f"Roll: {np.degrees(history[i][0]):.2f}° | "
        f"Pitch: {np.degrees(history[i][1]):.2f}° | "
        f"Yaw: {np.degrees(history[i][2]):.2f}°"
    )
    ax.set_title(title_text, color='white', fontsize=12)

# --- Animation Control (Move these OUTSIDE of update) ---
skip = 50  
ani = FuncAnimation(
    fig, 
    update, 
    frames=range(0, steps, skip), 
    interval=dt * 1000, 
    repeat=False
)
plt.show()

# USE THIS FOR STATIC PLOTS INSTEAD OF ANIMATION
history_np = np.array(history)
plt.figure(figsize=(10, 6))
plt.title("Pseudoinverse-based Control Tracking", fontsize=16)
plt.plot(time[:-1], np.degrees(history_np[:, 0]), alpha=0.8, lw=2.0, label="roll")
plt.plot(time[:-1], np.degrees(history_np[:, 1]), alpha=0.8, lw=2.0, label="pitch")
plt.plot(time[:-1], np.degrees(history_np[:, 2]), alpha=0.8, lw=2.0, label="yaw")

plt.xlabel("Time [s]", fontsize=14)
plt.ylabel("Angle [deg]", fontsize=14)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()