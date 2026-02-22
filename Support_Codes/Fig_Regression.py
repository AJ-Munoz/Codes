import matplotlib.pyplot as plt

# RGB colors (normalized to 0-1)
bg_color = (10/255, 10/255, 10/255)      # Dark Background
neongreen = (0/255, 180/255, 70/255)    # Green Lines
neonblue = (0/255, 200/255, 220/255)    # Blue Lines
neonpink = (220/255, 20/255, 150/255)  # Pink/Magenta Drawings

def draw_custom_neuron():
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=bg_color)
    ax.set_facecolor(bg_color)
    
    inputs_y = [4, 3, 1]  # Y positions for x1, x2, xn
    sum_x, sum_y = 6, 2.5 # Summation node center
    
    for i, y in enumerate(inputs_y):
        label_x = f"$x_{1}$" if i == 0 else (f"$x_{2}$" if i == 1 else "$x_n$")
        label_f = f"$f_{1}$" if i == 0 else (f"$f_{2}$" if i == 1 else "$f_n$")
        
        # 1. Inputs and Nonlinear Functions
        ax.plot(0.5, y, 'o', color=neongreen, markersize=7)
        ax.text(0.1, y, label_x, color=neongreen, fontsize=12, va='center')
        
        circle = plt.Circle((1.5, y), 0.35, fill=False, edgecolor=neongreen, lw=2)
        ax.add_patch(circle)
        ax.text(1.5, y, label_f, color=neongreen, ha='center', va='center')
        
        # 2. Connections
        ax.plot([0.6, 1.15], [y, y], color=neongreen, lw=1.5) # x to f
        ax.annotate('', xy=(sum_x - 0.45, sum_y + (y-sum_y)*0.1), xytext=(1.85, y), 
                    arrowprops=dict(arrowstyle="->", color=neongreen, lw=1.5))
        
        # 3. Weights (Parameters)
        w_label = f"$w_{1}$" if i == 0 else (f"$w_{2}$" if i == 1 else "$w_n$")
        ax.text(3.8, (y + sum_y)/2 + 0.1, w_label, color=neonpink, fontsize=11)

    # Dots for "..."
    ax.text(1.5, 2.1, "$\dots$", color=neongreen, ha='center', fontsize=20, rotation=90)

    # 4. Summation Node
    sum_circle = plt.Circle((sum_x, sum_y), 0.5, fill=False, edgecolor=neongreen, lw=2)
    ax.add_patch(sum_circle)
    ax.text(sum_x, sum_y, r"$\sum$", color=neongreen, fontsize=22, ha='center', va='center')

    # 5. FIXED: Bias line pointing DOWN to the summation node
    ax.annotate('', xy=(sum_x, sum_y + 0.5), xytext=(sum_x, 4.2), 
                arrowprops=dict(arrowstyle="->", color=neongreen, lw=1.5))
    ax.text(sum_x + 0.2, 3.8, "$w_{0}$", color=neonpink, fontsize=14)
    ax.text(sum_x, 4.5, "bias", color=neonblue, ha='center', fontsize=11)

    # 6. Output
    ax.annotate('', xy=(8.5, sum_y), xytext=(sum_x + 0.5, sum_y), 
                arrowprops=dict(arrowstyle="->", color=neongreen, lw=1.5))
    ax.text(8.7, sum_y, "$\hat{y}$", color=neongreen, fontsize=16, va='center')
    ax.text(8.7, sum_y - 0.6, "estimation", color=neonblue, ha='center', fontsize=11)

    # Additional text labels
    ax.text(0.3, 0.2, "features", color=neonblue, ha='center', fontsize=11)
    ax.text(1.7, 0.0, "Nonlinear\nFunctions", color=neonblue, ha='center', fontsize=10)
    ax.text(3.8, 4.6, "parameters", color=neonblue, ha='center', fontsize=10)

    ax.set_xlim(-0.5, 10)
    ax.set_ylim(-0.5, 5)
    plt.axis('off')
    plt.show()

draw_custom_neuron()
