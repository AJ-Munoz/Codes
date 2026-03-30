import numpy as np
import matplotlib.pyplot as plt

# Normalized RGB colors
bkg_rgb = (10/255, 10/255, 10/255)      # Black background
neongray_rgb = (180/255, 185/255, 195/255) # Neon-gray for the plane Im(P_T)
neonblue_rgb = (0/255, 200/255, 220/255) # Y-hat vector
neongreen_rgb = (0/255, 180/255, 70/255) # Y vector
magenta_rgb = (220/255, 20/255, 150/255) # Magenta for Im(Q_T) / Error E

def draw_projection_spaces():
    fig = plt.figure(figsize=(10, 8), facecolor=bkg_rgb)
    ax = fig.add_subplot(111, projection='3d', facecolor=bkg_rgb)

    # 1. Create the Tilted Plane Im(P_T)
    x = np.linspace(-2, 2, 12)
    y = np.linspace(-2, 2, 12)
    X, Y_grid = np.meshgrid(x, y)
    Z = 0.3 * X + 0.2 * Y_grid 

    # Draw the Im(P_T) mesh
    ax.plot_wireframe(X, Y_grid, Z, color=neongray_rgb, alpha=0.4, linewidth=1.0)
    ax.text(2.2, 2.2, 1.2, "$Im(P_T)$", color=neongray_rgb, fontsize=14, fontweight='bold')

    # 2. Vector Coordinates
    origin = np.array([0, 0, 0])
    y_hat = np.array([1.2, 1.2, 0.6])  # Projection on Im(P_T)
    error = np.array([-0.4, -0.4, 1.8]) # Error vector E (Normal to plane)
    y_vec = y_hat + error              # Original Vector Y

    # 3. Draw Vectors
    # Vector Y (Green) - The target vector
    ax.quiver(*origin, *y_vec, color=neongreen_rgb, linewidth=3, arrow_length_ratio=0.1)
    ax.text(*(y_vec * 1.1), "$Y$", color=neongreen_rgb, fontsize=16, fontweight='bold')

    # Vector Y-hat (Blue) - Lies in Im(P_T)
    ax.quiver(*origin, *y_hat, color=neonblue_rgb, linewidth=3, arrow_length_ratio=0.1)
    ax.text(*(y_hat * 1.1), "$\hat{Y} \in Im(P_T)$", color=neonblue_rgb, fontsize=14, fontweight='bold')

    # Error Vector E (Magenta) - Lives in Im(Q_T)
    # We draw it at the origin to show it spans the normal space, and at y_hat to show the distance
    ax.quiver(*origin, *error, color=magenta_rgb, linewidth=2, linestyle='-', arrow_length_ratio=0.1)
    ax.text(*(error * 1.1), "$Im(Q_T)$", color=magenta_rgb, fontsize=14, fontweight='bold')
    
    # Connecting line for the residual (orthogonal distance)
    ax.plot([y_hat[0], y_vec[0]], [y_hat[1], y_vec[1]], [y_hat[2], y_vec[2]], 
            color=magenta_rgb, linestyle='--', linewidth=2)
    ax.text(*(y_hat + error/2 + 0.2), "$E$", color=magenta_rgb, fontsize=16, fontweight='bold')

    # 4. Final Styling
    ax.set_axis_off()
    ax.view_init(elev=20, azim=-60)
    ax.set_xlim([-2.5, 2.5]); ax.set_ylim([-2.5, 2.5]); ax.set_zlim([0, 3])
    
    plt.show()

if __name__ == "__main__":
    draw_projection_spaces()
