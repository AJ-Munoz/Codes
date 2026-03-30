import matplotlib.pyplot as plt

# Neon color palette
C = {
    'bg':    (10/255, 10/255, 10/255),
    'blue':  (0/255, 200/255, 220/255),    # Layer 1
    'gold':  (210/255, 140/255, 0/255),    # Layer 2
    'white': (240/255, 240/255, 240/255),  # Layer 3
    'pink':  (220/255, 20/255, 150/255),   # Layer 4 (Adaptive)
    'green': (50/255, 255/255, 50/255),    # Layer 5
    'gray':  (170/255, 170/255, 170/255)
}

def draw_anfis_architecture():
    fig, ax = plt.subplots(figsize=(14, 8), facecolor=C['bg'])
    ax.set_facecolor(C['bg'])

    # Layer configuration: (Label, number of nodes, color, internal symbol)
    layers = [
        ("Layer 1:\nFuzzification", 4, C['blue'],  "$\mu$"),
        ("Layer 2:\nRule Firing",   4, C['gold'],  "$\Pi$"),
        ("Layer 3:\nNormalization", 4, C['white'], "$N$"),
        ("Layer 4:\nLocal TS",      4, C['pink'],  "$f_i$"),
        ("Layer 5:\nAggregation",   1, C['green'], "$\Sigma$")
    ]

    x_spacing = 2.8
    r = 0.35
    layer_nodes = []

    # Draw Nodes
    for i, (label, n_nodes, color, sym) in enumerate(layers):
        x = i * x_spacing
        y_vals = [(k - (n_nodes - 1) / 2) for k in range(n_nodes)]
        nodes = []

        for y in y_vals:
            # Highlight Layer 4 (Adaptive) with a thicker edge
            lw = 3.5 if i == 3 else 1.5
            circ = plt.Circle((x, y), r, color=color, ec='white', lw=lw, zorder=5)
            ax.add_patch(circ)
            
            # Add mathematical symbol inside node
            txt_color = 'black' if color in [C['white'], C['green']] else 'white'
            ax.text(x, y, sym, color=txt_color, ha='center', va='center', 
                    fontsize=11, fontweight='bold', zorder=6)
            nodes.append((x, y))

        layer_nodes.append(nodes)
        ax.text(x, 3.0, label, color=C['gray'], ha='center', va='bottom', 
                fontsize=10, fontweight='bold')

    # Draw Connections
    for i in range(len(layer_nodes) - 1):
        for (x1, y1) in layer_nodes[i]:
            for (x2, y2) in layer_nodes[i+1]:
                ax.plot([x1 + r, x2 - r], [y1, y2], color=C['gray'], alpha=0.2, lw=1, zorder=1)

    # Input arrows
    for (x, y) in layer_nodes[0]:
        ax.annotate("", xy=(x - r, y), xytext=(x - 1.2, y),
                    arrowprops=dict(arrowstyle="->", color=C['blue'], lw=2))

    # Output arrow
    x_out, y_out = layer_nodes[-1][0]
    ax.annotate("", xy=(x_out + 1.5, y_out), xytext=(x_out + r, y_out),
                arrowprops=dict(arrowstyle="->", color=C['green'], lw=2.5))
    ax.text(x_out + 1.6, y_out, "OUTPUT", color=C['green'], fontweight='bold', va='center')

    # Adaptation Highlight for Layer 4
    ax.text(3 * x_spacing, -3.0, "ADAPTIVE PARAMETERS", color=C['pink'], 
            ha='center', fontsize=10, 
            bbox=dict(facecolor='none', edgecolor=C['pink'], pad=5, lw=2))

    ax.set_xlim(-2, x_spacing * 4 + 3)
    ax.set_ylim(-4, 4)
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.title("Adaptive Neuro-Fuzzy Inference System (ANFIS) Architecture", 
              color='white', fontsize=15, pad=30)
    plt.show()

if __name__ == "__main__":
    draw_anfis_architecture()
import matplotlib.pyplot as plt

# Neon color palette
C = {
    'bg':    (10/255, 10/255, 10/255),
    'blue':  (0/255, 200/255, 220/255),    # Layer 1
    'gold':  (210/255, 140/255, 0/255),    # Layer 2
    'white': (240/255, 240/255, 240/255),  # Layer 3
    'pink':  (220/255, 20/255, 150/255),   # Layer 4 (Adaptive)
    'green': (50/255, 255/255, 50/255),    # Layer 5
    'gray':  (170/255, 170/255, 170/255)
}

def draw_anfis_architecture():
    fig, ax = plt.subplots(figsize=(14, 8), facecolor=C['bg'])
    ax.set_facecolor(C['bg'])

    # Layer configuration: (Label, number of nodes, color, internal symbol)
    layers = [
        ("Layer 1:\nFuzzification", 4, C['blue'],  "$\mu$"),
        ("Layer 2:\nRule Firing",   4, C['gold'],  "$\Pi$"),
        ("Layer 3:\nNormalization", 4, C['white'], "$N$"),
        ("Layer 4:\nLocal TS",      4, C['pink'],  "$f_i$"),
        ("Layer 5:\nAggregation",   1, C['green'], "$\Sigma$")
    ]

    x_spacing = 2.8
    r = 0.35
    layer_nodes = []

    # Draw Nodes
    for i, (label, n_nodes, color, sym) in enumerate(layers):
        x = i * x_spacing
        y_vals = [(k - (n_nodes - 1) / 2) for k in range(n_nodes)]
        nodes = []

        for y in y_vals:
            # Highlight Layer 4 (Adaptive) with a thicker edge
            lw = 3.5 if i == 3 else 1.5
            circ = plt.Circle((x, y), r, color=color, ec='white', lw=lw, zorder=5)
            ax.add_patch(circ)
            
            # Add mathematical symbol inside node
            txt_color = 'black' if color in [C['white'], C['green']] else 'white'
            ax.text(x, y, sym, color=txt_color, ha='center', va='center', 
                    fontsize=11, fontweight='bold', zorder=6)
            nodes.append((x, y))

        layer_nodes.append(nodes)
        ax.text(x, 3.0, label, color=C['gray'], ha='center', va='bottom', 
                fontsize=10, fontweight='bold')

    # Draw Connections
    for i in range(len(layer_nodes) - 1):
        for (x1, y1) in layer_nodes[i]:
            for (x2, y2) in layer_nodes[i+1]:
                ax.plot([x1 + r, x2 - r], [y1, y2], color=C['gray'], alpha=0.2, lw=1, zorder=1)

    # Input arrows
    for (x, y) in layer_nodes[0]:
        ax.annotate("", xy=(x - r, y), xytext=(x - 1.2, y),
                    arrowprops=dict(arrowstyle="->", color=C['blue'], lw=2))

    # Output arrow
    x_out, y_out = layer_nodes[-1][0]
    ax.annotate("", xy=(x_out + 1.5, y_out), xytext=(x_out + r, y_out),
                arrowprops=dict(arrowstyle="->", color=C['green'], lw=2.5))
    ax.text(x_out + 1.6, y_out, "OUTPUT", color=C['green'], fontweight='bold', va='center')

    # Adaptation Highlight for Layer 4
    ax.text(3 * x_spacing, -3.2, "ADAPTIVE PARAMETERS", color=C['pink'], 
            ha='center', fontsize=11, fontweight='bold', 
            bbox=dict(facecolor='none', edgecolor=C['pink'], pad=5, lw=2))

    ax.set_xlim(-2, x_spacing * 4 + 3)
    ax.set_ylim(-4, 4)
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.title("Adaptive Neuro-Fuzzy Inference System (ANFIS) Architecture", 
              color='white', fontsize=15, pad=30)
    plt.show()

if __name__ == "__main__":
    draw_anfis_architecture()
