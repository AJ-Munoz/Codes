#Plot Data
# Load data: t, e, theta, ref, u, dt
import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('data.txt')
t, e, theta, ref, u, dt = data.T

# Calculate cumulative metrics for plotting
ise = np.cumsum(dt * e**2)
isc = np.cumsum(dt * u**2)

fig, axs = plt.subplots(2, 2, figsize=(10, 7), sharex=True)

# State tracking
axs[0][0].plot(t, ref, color=[0.6, 0.9, 0.7], label='Reference [rad]', linewidth=6)
axs[0][0].plot(t, theta, 'b--', label='Angle [rad]')
axs[0][0].set_ylabel('Value')
axs[0][0].legend(fontsize=14)
axs[0][0].grid(True)  # Added grid for 1st plot

# Control effort
axs[0][1].plot(t, u, 'g-', label='Control ($u$)')
axs[0][1].set_ylabel('Control Signal')
axs[0][1].legend(fontsize=14)
axs[0][1].grid(True)  # Added grid for 2nd plot

# Performance metrics
axs[1][0].plot(t, ise, 'g-', label='ISE')
axs[1][0].set_xlabel('Time (s)')
axs[1][0].set_ylabel('Cumulative Cost')
axs[1][0].legend(fontsize=14)
axs[1][0].grid(True)  # Added grid for 3rd plot

# Performance metrics
axs[1][1].plot(t, isc, 'g-', label='ISC')
axs[1][1].set_xlabel('Time (s)')
axs[1][1].set_ylabel('Cumulative Cost')
axs[1][1].legend(fontsize=14)
axs[1][1].grid(True)  # Added grid for 4th plot

plt.tight_layout()
plt.show()
