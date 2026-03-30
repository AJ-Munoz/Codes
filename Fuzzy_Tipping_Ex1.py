import numpy as np
import skfuzzy as fuzz
# pip install scikit-fuzzy                                                                             

# -------------------------------------------------
# Input Range 
# -------------------------------------------------
x = np.arange(0, 11, 1) # 0 to 10 inclusive

# -------------------------------------------------
# Membership Functions 
# -------------------------------------------------
service = {
    'poor':    fuzz.trimf(x, [0, 0, 5]),  # 0 to 5, peaking at 0
    'average': fuzz.trimf(x, [0, 5, 10]), # 0 to 10, peaking at 5
    'good':    fuzz.trimf(x, [5, 10, 10]) # 5 to 10, peaking at 10
}

quality = {
    'poor':    fuzz.trimf(x, [0, 0, 5]),  # 0 to 5, peaking at 0
    'average': fuzz.trimf(x, [0, 5, 10]), # 0 to 10, peaking at 5
    'good':    fuzz.trimf(x, [5, 10, 10]) # 5 to 10, peaking at 10
}

# -------------------------------------------------
# Card Input Values (Service and Food Quality)
# -------------------------------------------------
service_in = 9.8
quality_in = 6.5

# -------------------------------------------------
# Rule Strengths (Using MAX for OR)
# -------------------------------------------------
low_strength = max(
    fuzz.interp_membership(x, service['poor'], service_in), 
    fuzz.interp_membership(x, quality['poor'], quality_in)
)

average_strength = max(
    fuzz.interp_membership(x, service['average'], service_in),
    fuzz.interp_membership(x, quality['average'], quality_in)
)

high_strength = max(
    fuzz.interp_membership(x, service['good'], service_in),
    fuzz.interp_membership(x, quality['good'], quality_in)
)

# -------------------------------------------------
# Takagi-Sugeno Outcomes (Tip Percentages)
# -------------------------------------------------
LOW_TIP     = 5
AVERAGE_TIP = 10
HIGH_TIP    = 15

# -------------------------------------------------
# Weighted Average Defuzzification
# -------------------------------------------------
weights_sum = low_strength + average_strength + high_strength

if weights_sum == 0:
    final_tip = AVERAGE_TIP  # Default to average if no rules fire
else:
    final_tip = (
        low_strength     * LOW_TIP + 
        average_strength * AVERAGE_TIP +
        high_strength    * HIGH_TIP
    ) / weights_sum

print(f"Final Tip: {final_tip:.2f}%") 