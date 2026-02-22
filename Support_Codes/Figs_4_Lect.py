import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-10, 10, 1000)
y = 1 / (1 + np.exp(-x))

# RGB (10, 10, 10) normalized for Matplotlib
bkg_rgb = (10/255, 10/255, 10/255)
neonblue_rgb = (0/255, 200/255, 220/255)
neongreen_rgb = (0/255, 180/255, 70/255)

fig, ax = plt.subplots(facecolor=bkg_rgb)
ax.set_facecolor(bkg_rgb)

# Plotting the neon blue line
plt.plot(x, y, color=neonblue_rgb, linewidth=4)

# Styling
plt.title("Sigmoid Function", color=neonblue_rgb, fontsize=16, pad=10)
plt.xlabel("X-axis", color=neonblue_rgb, fontsize=14)
plt.ylabel("Y-axis", color=neonblue_rgb, fontsize=14)

ax.tick_params(colors=neongreen_rgb)
for spine in ax.spines.values():
    spine.set_color("#555454")

plt.grid(color='#555454')
plt.show()
