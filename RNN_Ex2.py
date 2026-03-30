import numpy as np
import tensorflow as tf
import os

# 1. Generate Data
btc_returns = np.array([ # Real BTC returns
    0.0, -0.0237, -0.0711, -0.0611, -0.0449, -0.0381, -0.0574, -0.0377, -0.0510, -0.0560,
    -0.0540, -0.0245, 0.0017, -0.0158, -0.0220, -0.0037, 0.0303, 0.0301, 0.0288, -0.0041,
    -0.0263, -0.0285, -0.0514, -0.0379, -0.1035, -0.1448, -0.1475, -0.2341, -0.2343, -0.2315,
    -0.2690, -0.2817, -0.2404, -0.2593, -0.2766, -0.2660, -0.2982, -0.2573, -0.2890, -0.2771,
    -0.2658, -0.2359, -0.2788, -0.2484, -0.2484, -0.2303, -0.2034, -0.1940, -0.2368, -0.2331,
    0.012, -0.008, 0.011, 0.007, 0.081, 0.048, 0.011, -0.010, -0.002, 0.031,
    0.033, 0.031, 0.059, 0.055, 0.059, 0.092, 0.105, 0.105, 0.038, 0.047
]) #

# 2. Memory Prep and Encoding 
window_size = 10
X, y = [], []

for i in range(len(btc_returns) - window_size):
    X.append(btc_returns[i : i + window_size])
    y.append(btc_returns[i + window_size])

X = np.array(X).reshape(-1, window_size, 1)
y = np.array(y)

# 3. Model Architecture and Training: Teaching Skynet to Predict the Markets
model = tf.keras.Sequential([
    #tf.keras.layers.SimpleRNN(32, input_shape=(window_size, 1), activation='tanh'),
    tf.keras.layers.LSTM(32, input_shape=(window_size, 1), activation='tanh'),
    tf.keras.layers.Dense(16, activation='tanh'), 
    tf.keras.layers.Dense(1) # Linear output for the final % prediction
])
model.compile(optimizer='adam', loss='mse')

os.system('cls' if os.name == 'nt' else 'clear')
print("Training Skynet on the Markets")
model.fit(X, y, epochs=50, verbose=0)

# 4. Next Move Prediction
last_window = btc_returns[-window_size:].reshape(1, window_size, 1)
next_move = model.predict(last_window, verbose=0)[0][0]

print(f"Recent Trend: {np.round(btc_returns[-3:], 4)}")
print(f"Predicted Move: {next_move:+.2%}")

if next_move > 5e-3: 
    print("To the moon! 🚀🌙")
elif next_move < -5e-3:
    print("Verdict: CRASH IMMINENT! 💥📉")
else:
    print("Verdict: HODL (Hold on for dear life). 🔒")
