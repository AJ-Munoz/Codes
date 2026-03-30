import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz

# -------------------------------------------------
# Neon color palette
# -------------------------------------------------
C = {
    'bg':    (10/255, 10/255, 10/255),
    'blue':  (0/255, 200/255, 220/255),
    'gray':  (170/255, 170/255, 170/255),
    'gold':  (210/255, 140/255, 0/255),
    'pink':  (220/255, 20/255, 150/255),
    'green': (50/255, 255/255, 50/255)
}

# -------------------------------------------------
# Domain
# -------------------------------------------------
theta_univ = np.linspace(0, 2*np.pi, 1000)
sin_true = np.sin(theta_univ)

def get_fuzzy_taylor_ts(n_rules, sigma_factor=1.0):
    centers = np.linspace(0, 2*np.pi, n_rules)
    # sigma logic for smooth blending
    dist = (2 * np.pi) / (n_rules - 1) if n_rules > 1 else 1.0
    sigma = dist * sigma_factor
    
    # First-order Taylor: f(x) ≈ sin(c) + cos(c)*(x - c)
    # Becomes: ax + b where a = cos(c) and b = sin(c) - c*cos(c)
    a = np.cos(centers)
    b = np.sin(centers) - centers * np.cos(centers)
    
    y_out = np.zeros_like(theta_univ)
    for i, val in enumerate(theta_univ):
        # Calculate membership for each rule
        mu = np.array([fuzz.gaussmf(val, c, sigma) for c in centers])
        mu_sum = np.sum(mu)
        
        if mu_sum > 1e-9:
            weights = mu / mu_sum
            # Linear rule consequence: a*x + b
            rule_outputs = a * val + b
            y_out[i] = np.sum(weights * rule_outputs)
        else:
            y_out[i] = np.sin(val)
    return y_out

# -------------------------------------------------
# Visualization
# -------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 6), facecolor=C['bg'])
ax.set_facecolor(C['bg'])

# Plot True Sine (The target)
ax.plot(theta_univ, sin_true, color=C['gray'], lw=10, label='True sin(θ)', alpha=0.3)

# 4 Rules (The "Rough" Tangent Blending)
ax.plot(theta_univ, get_fuzzy_taylor_ts(4), color=C['pink'], lw=2, 
        ls='--', label='TS Taylor (4 rules)')

# 16 Rules (The High Accuracy Sweet Spot)
ax.plot(theta_univ, get_fuzzy_taylor_ts(16), color=C['gold'], lw=2, 
        ls='-.', label='TS Taylor (16 rules)')

# 64 Rules (Overkill / Perfect Precision)
ax.plot(theta_univ, get_fuzzy_taylor_ts(64), color=C['blue'], lw=1.5, 
        ls=':', alpha=0.9, label='TS Taylor (64 rules)')

# Styling the Neon Grid
ax.set_title("First-Order Taylor–TS Approximation of sin(θ)", color='white', fontsize=15, pad=20)
ax.set_xlabel("θ [rad]", color=C['gray'])
ax.set_ylabel("f(θ)", color=C['gray'])
ax.tick_params(colors=C['gray'])
ax.grid(True, color=(0.15, 0.15, 0.15), linestyle='-')

# Legend setup
leg = ax.legend(facecolor=C['bg'], edgecolor=C['gray'])
for text in leg.get_texts():
    text.set_color('white')

plt.tight_layout()
plt.show()
