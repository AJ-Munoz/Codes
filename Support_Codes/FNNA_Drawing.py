import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

def draw_final_matrix_nn():
    # Background color
    bg_color = (10/255, 10/255, 10/255)
    
    # Custom Color Definitions
    url_gold, matrix_cyan, neon_pink = (210/255, 140/255, 0/255), (0/255, 200/255, 220/255), (220/255, 20/255, 150/255)
    matrix_green, silver = (0/255, 230/255, 90/255), (192/255, 192/255, 192/255)

    fig, ax = plt.subplots(figsize=(16, 9), dpi=200) 
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    
    layers = [
        {'pos': 1, 'n': 3, 'color': matrix_cyan, 'label': 'x', 'title': 'Input Layer'},
        {'pos': 3, 'n': 3, 'color': url_gold, 'label': 'W', 'title': 'Hidden Layer'},
        {'pos': 5, 'n': 1, 'color': neon_pink, 'label': 'ŷ', 'title': 'Output Layer'}
    ]
    
    node_coords = []
    for layer in layers:
        y_positions = [(j - (layer['n']-1)/2) for j in range(layer['n'])]
        layer_coords = []
        for y in y_positions:
            circle = plt.Circle((layer['pos'], y), 0.18, color=layer['color'], ec=silver, lw=2.5, zorder=4, antialiased=True)
            ax.add_artist(circle)
            
            label = f"$x_{int(y + (layer['n']-1)/2 + 1)}$" if layer['label'] == 'x' else f"${layer['label']}$"
            ax.text(layer['pos'], y, label, ha='center', va='center', color='white', fontsize=14, zorder=5)
            layer_coords.append((layer['pos'], y))
        node_coords.append(layer_coords)
        ax.text(layer['pos'], -1.8, layer['title'], ha='center', color=layer['color'], fontsize=16)

    # --- ADDED: Input Arrows pointing to the first layer ---
    for node in node_coords[0]:
        ax.annotate("", xy=node, xytext=(node[0] - 0.6, node[1]), 
                    arrowprops=dict(arrowstyle="->", color=matrix_green, lw=2.0, shrinkB=15, antialiased=True), zorder=1)

    # Connections between layers
    for i in range(len(node_coords) - 1):
        for start in node_coords[i]:
            for end in node_coords[i+1]:
                ax.annotate("", xy=end, xytext=start, 
                            arrowprops=dict(arrowstyle="->", color=matrix_green, lw=1.5, alpha=0.8, shrinkA=15, shrinkB=15, antialiased=True), zorder=1)

    # Prediction Arrow
    ax.annotate("", xy=(5.8, 0), xytext=(5.2, 0), 
                arrowprops=dict(arrowstyle="->", color=matrix_green, lw=2.0, antialiased=True), zorder=1)

    # Ground Truth Node
    ax.add_artist(plt.Circle((6.2, 0.6), 0.15, color=silver, alpha=0.4, ec=silver, lw=1.0, zorder=4, antialiased=True))
    ax.text(6.2, 0.6, "$y$", ha='center', va='center', color='white', fontsize=12)

    # Descriptive Labels
    ax.text(1, 1.5, "1. Inputs enter\nthe input layer", ha='center', color='white', fontsize=10, linespacing=1.3)
    ax.text(3, 1.5, "2. Weights ($W$)\nare applied", ha='center', color='white', fontsize=10, linespacing=1.3)
    ax.text(5, 0.5, "3. $\hat{y}$ is predicted", ha='center', color='white', fontsize=10)

    # Loss Section
    ax.text(6.4, 0.3, "4. Loss Calculation", ha='left', va='center', fontsize=10, color=matrix_green)
    ax.text(6.4, 0.05, "$L = (\hat{y} - y)^2$", ha='left', va='center', fontsize=15, color='white')
    ax.text(6.4, -0.3, "Difference between\npredicted & actual", ha='left', va='center', fontsize=11, color='silver')

    ax.set_xlim(0, 8.5)
    ax.set_ylim(-2.2, 2.2)
    ax.set_axis_off()
    
    plt.tight_layout()
    plt.show()

draw_final_matrix_nn()
