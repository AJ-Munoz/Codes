import os
# Hides INFO and WARNING logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np

# Reproducibility NumPy and TensorFlow seeds 
np.random.seed(42)
tf.random.set_seed(42)

# XOR data
X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32)
y = np.array([[0],[1],[1],[0]], dtype=np.float32)

# Initializer to match numpy: uniform(-1, 1) with seed
uinit = tf.keras.initializers.RandomUniform(minval=-1.0, maxval=1.0, seed=42)

# Build model: 2 -> 2 -> 1, sigmoid activations, same init for kernels & biases
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2,)), 
    tf.keras.layers.Dense(2, activation='sigmoid',
                          kernel_initializer=uinit, bias_initializer=uinit),
    
    tf.keras.layers.Dense(1, activation='sigmoid',
                          kernel_initializer=uinit, bias_initializer=uinit)
])


# Loss with SUM reduction to match numpy's batch-sum gradient scaling
loss_sum_mse = tf.keras.losses.MeanSquaredError(
    reduction=tf.keras.losses.Reduction.SUM
)

# SGD with lr=0.5 (matches your numpy lr and update style)
opt = tf.keras.optimizers.SGD(learning_rate=0.5)

model.compile(optimizer=opt, loss=loss_sum_mse, metrics=['binary_accuracy'])

# Train with full-batch updates (batch_size=4), no shuffling
model.fit(X, y, epochs=1000, batch_size=4, shuffle=False, verbose=0)

# Predictions
pred = model.predict(X, verbose=0)
print("Final Predictions (raw):\n", np.round(pred, 6))
print("Rounded Predictions:\n", np.round(pred))