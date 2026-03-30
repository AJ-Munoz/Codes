import numpy as np
import tensorflow as tf
import os
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Create and Load Dataset

# New Dataset
Dataset = {
    'x1': [
        0, 1, 0, 1, 0, 1, 0, 1 # Input 1
    ],
    'x2': [
        0, 0, 1, 1, 0, 0, 1, 1 # Input 2
    ],
    'x3': [
        0, 0, 0, 0, 1, 1, 1, 1 # Input 3
    ],
    'label': [
        0, 1, 1, 0, 1, 0, 0, 1 # Output (XOR of the three inputs)
    ]
}

df = pd.DataFrame(Dataset)
X = df[['x1', 'x2', 'x3']].values
y = df['label'].values

if os.path.exists('triple_xor_model.keras'):
    model = tf.keras.models.load_model('triple_xor_model.keras')
else:
    # Create Neural Network
    model = Sequential([
        Dense(64, input_dim=3, activation='relu'),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    # Compile the Model
    model.compile(
        loss='binary_crossentropy', 
        optimizer='adam',
        metrics=['accuracy']
    )

    # Train the Model
    model.fit(X, y, epochs=100, batch_size=1, verbose=0)
    model.save('triple_xor_model.keras')

# Print Predictions
pred = model.predict(X, verbose=0)
print("Final Predictions (raw):\n", np.round(pred, 6))
print("Rounded Predictions:\n", np.round(pred))
