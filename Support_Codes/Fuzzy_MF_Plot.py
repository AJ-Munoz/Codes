import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

# -------------------------------------------------
# Neon color palette (normalized RGB)
# -------------------------------------------------
C = {
    'bg':    (10/255, 10/255, 10/255),
    'blue':  (0/255, 200/255, 220/255),
    'gray':  (150/255, 150/255, 150/255),
    'gold':  (210/255, 140/255, 0/255),
    'pink':  (220/255, 20/255, 150/255),
}

# -------------------------------------------------
# Global matplotlib style (black background)
# -------------------------------------------------
plt.rcParams.update({
    'figure.facecolor': C['bg'],
    'axes.facecolor':   C['bg'],
    'savefig.facecolor': C['bg'],
    'axes.edgecolor':   C['gray'],
    'axes.labelcolor':  C['gray'],
    'xtick.color':      C['gray'],
    'ytick.color':      C['gray'],
    'text.color':       C['gray'],
})

# -------------------------------------------------
# Input universe
# -------------------------------------------------
x = np.arange(0, 11, 1)

# -------------------------------------------------
# Fuzzy membership functions (inputs only)
# -------------------------------------------------
quality = {
    'poor':    fuzz.trimf(x, [0, 0, 5]),
    'average': fuzz.trimf(x, [0, 5, 10]),
    'good':    fuzz.trimf(x, [5, 10, 10])
}

service = {
    'poor':    fuzz.trimf(x, [0, 0, 5]),
    'average': fuzz.trimf(x, [0, 5, 10]),
    'good':    fuzz.trimf(x, [5, 10, 10])
}

# -------------------------------------------------
# Plot QUALITY membership functions (neon)
# -------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(x, quality['poor'],    color=C['gold'], linewidth=2.5, label='Poor')
ax.plot(x, quality['average'], color=C['blue'], linewidth=2.5, label='Average')
ax.plot(x, quality['good'],    color=C['pink'], linewidth=2.5, label='Good')

ax.set_title("Quality Membership Functions", color=C['blue'], fontsize=13)
ax.set_xlabel("Quality (0–10)", color=C['gray'])
ax.set_ylabel("Membership Degree", color=C['gray'])
ax.set_ylim(0, 1.05)
ax.tick_params(colors=C['gray'])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color(C['gray'])
ax.spines['bottom'].set_color(C['gray'])
ax.legend(facecolor=C['bg'], edgecolor=C['gray'], labelcolor=C['gray'])

fig.tight_layout()
fig.savefig("membership.png", dpi=200)
plt.close(fig)

# -------------------------------------------------
# Example input (revealed slide)
# -------------------------------------------------
service_in = 9.8
quality_in = 6.5

# -------------------------------------------------
# Rule firing strengths (fuzzy premises)
# -------------------------------------------------
w1 = max(
    fuzz.interp_membership(x, service['poor'], service_in),
    fuzz.interp_membership(x, quality['poor'], quality_in)
)

w2 = fuzz.interp_membership(x, service['average'], service_in)

w3 = max(
    fuzz.interp_membership(x, service['good'], service_in),
    fuzz.interp_membership(x, quality['good'], quality_in)
)

weights = np.array([w1, w2, w3])
labels = ['LOW', 'MEDIUM', 'HIGH']

# -------------------------------------------------
# Zero‑order Takagi–Sugeno consequents
# -------------------------------------------------
tips = np.array([5, 10, 15])  # percent

# -------------------------------------------------
# Weighted average (NO Mamdani, NO defuzzification)
# -------------------------------------------------
final_tip = np.dot(weights, tips) / np.sum(weights)

print(f"Final Tip: {final_tip:.2f}%")

# -------------------------------------------------
# Visualization: rule contributions + final result
# -------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(7, 4))

ax2.bar(labels, weights, color=[C['gold'], C['blue'], C['pink']])
ax2.set_ylim(0, 1.05)
ax2.set_ylabel("Rule Strength", color=C['gray'])
ax2.set_title("Rule Activations", color=C['blue'], fontsize=13)

ax2.tick_params(colors=C['gray'])
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_color(C['gray'])
ax2.spines['bottom'].set_color(C['gray'])

fig2.tight_layout()
fig2.savefig("rules.png", dpi=200)
plt.close(fig2)

# -------------------------------------------------
# Visualization: final crisp output only
# -------------------------------------------------
fig3, ax3 = plt.subplots(figsize=(7, 2))
ax3.axvline(final_tip, color=C['gray'], linewidth=3)
ax3.set_xlim(0, 20)
ax3.set_yticks([])
ax3.set_xlabel("Tip (%)", color=C['gray'])
ax3.set_title("Final Tip (Weighted Average)", color=C['blue'], fontsize=13)

ax3.tick_params(colors=C['gray'])
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['left'].set_visible(False)
ax3.spines['bottom'].set_color(C['gray'])

fig3.tight_layout()
fig3.savefig("result.png", dpi=200)
plt.close(fig3)