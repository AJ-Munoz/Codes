import matplotlib.pyplot as plt
import numpy as np

def draw_matrix_perceptron():
    # Background color
    bg_color = (10/255, 10/255, 10/255)
    
    # Custom Color Definitions
    url_gold, matrix_cyan, neon_pink = (210/255, 140/255, 0/255), (0/255, 200/255, 220/255), (220/255, 20/255, 150/255)
    matrix_green, silver = (0/255, 230/255, 90/255), (192/255, 192/255, 192/255)

    fig, ax = plt.subplots(figsize=(16, 8), dpi=200) 
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    
    # Node Positions
    input_x = 1
    input_y = [1.5, 0.5, -1.5] # Positions for x1, x2, xn
    sum_node_pos = (4, 0)
    act_node_pos = (7, 0)
    output_x = 9.5

    # 1. Draw Input Layer (x1, x2, xn)
    labels = ['$x_1$', '$x_2$', '$x_n$']
    for i, y in enumerate(input_y):
        # Input Arrows (from left into x nodes)
        ax.annotate("", xy=(input_x - 0.32, y), xytext=(input_x - 0.8, y), 
                    arrowprops=dict(arrowstyle="->", color=matrix_green, lw=1.5, antialiased=True))
        
        # Nodes
        circle = plt.Circle((input_x, y), 0.35, color=matrix_cyan, ec=silver, lw=2.5, zorder=4)
        ax.add_artist(circle)
        ax.text(input_x, y, labels[i], ha='center', va='center', color='white', fontsize=18, zorder=5)
        
        # Weights (w1, w2, wn) and Connections
        ax.annotate("", xy=(sum_node_pos[0]-0.48, sum_node_pos[1]), xytext=(input_x + 0.3, y), 
                    arrowprops=dict(arrowstyle="->", color=matrix_green, lw=1.5, alpha=0.8))
        
        weight_label = f"$w_{{{'1' if i==0 else '2' if i==1 else 'n'}}}$"
        # Position weight text along the lines
        tx, ty = (input_x + sum_node_pos[0])/2, (y + sum_node_pos[1])/2 + 0.15
        ax.text(tx, ty, weight_label, color='white', fontsize=14, weight='bold')

    # Dotted line for inputs
    ax.text(input_x, -0.5, "...", rotation=90, ha='center', color=silver, fontsize=25)
    ax.text(input_x, -2.2, "Inputs", ha='center', color=matrix_cyan, fontsize=16)
    ax.text(input_x + 1.7, -1.7, "Weights", ha='center', color=silver, fontsize=12)

    # 2. Weighted Sum Node (Σ)
    ax.add_artist(plt.Circle(sum_node_pos, 0.5, color=url_gold, ec=silver, lw=2.5, zorder=4))
    ax.text(sum_node_pos[0], sum_node_pos[1], "$\Sigma$", ha='center', va='center', color='white', fontsize=24, zorder=5)
    ax.text(sum_node_pos[0], 0.8, "Weighted\nSum", ha='center', color=url_gold, fontsize=16, linespacing=1.2)

    # 3. Activation Function Node (Integral/Sigmoid style)
    ax.add_artist(plt.Circle(act_node_pos, 0.5, color=url_gold, ec=silver, lw=2.5, zorder=4))
    ax.text(act_node_pos[0], act_node_pos[1], "$f( )$", ha='center', va='center', color='white', fontsize=24, zorder=5)
    ax.text(act_node_pos[0], 0.8, "Activation\nFunction", ha='center', color=url_gold, fontsize=16, linespacing=1.2)

    # Connection: Sum -> Activation
    ax.annotate("", xy=(act_node_pos[0]-0.5, 0), xytext=(sum_node_pos[0]+0.5, 0), 
                arrowprops=dict(arrowstyle="->", color=matrix_green, lw=2.0))

    # 4. Output Layer
    ax.annotate("", xy=(output_x - 0.5, 0), xytext=(act_node_pos[0]+0.5, 0), 
                arrowprops=dict(arrowstyle="->", color=matrix_green, lw=2.0))
    ax.text(output_x, 0, "$\hat{y}$", ha='center', va='center', color='white', fontsize=20)
    ax.text(output_x, 0.5, "Output", ha='center', color=neon_pink, fontsize=16)

    ax.set_xlim(-0.5, 11)
    ax.set_ylim(-3, 3)
    ax.set_axis_off()
    
    plt.tight_layout()
    plt.show()

draw_matrix_perceptron()
