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

# Setup simulation
dt, steps = 0.001, 10000
eta = np.array([0.0, 1.5708, 0.0])  # Initial orientation (roll, pitch, yaw)
omega_cmd = np.array([0.1, 0.05, 0.2])
history = []
time = [0]

for i in range(steps):
    W_inv = damped_inverse(get_W(eta[0], eta[1]), 0.1)
    omega_cmd += 0.01 * (np.random.rand(3) - 0.5)  # Add small random noise to the command
    eta += (W_inv @ omega_cmd) * dt
    history.append(eta.copy())
    time.append(time[-1] + dt)

# --- Pre-generate Stars ---
num_stars = 500
star_coords = (np.random.rand(num_stars, 3) - 0.5) * 8 
star_sizes = np.random.rand(num_stars) * 7           

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
        f"Orbital Tracking (Gemini Generated)\n"
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
