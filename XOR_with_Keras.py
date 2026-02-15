import tensorflow as tf
import numpy as np

# 1. Data (XOR)
X = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
y = np.array([[0], [1], [1], [0]], dtype=np.float32)

# 2. Build Model
model = tf.keras.Sequential([
    # Increase to 4 neurons for reliable convergence
    tf.keras.layers.Dense(32, input_dim=2, activation='tanh'), 
    # Sigmoid is required at the output for binary classification
    tf.keras.layers.Dense(1, activation='sigmoid') 
])

# 3. Use Adam Optimizer and Mean Squared Error or Binary Crossentropy
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])

# 4. Train (epochs must be high enough, ~500-1000)
model.fit(X, y, epochs=100, verbose=0)

# 5. Result
print(model.predict(X).round())
