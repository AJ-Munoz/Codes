import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Create and Load Dataset
"""
data = {
    'feature1': [0.1, 0.2, 0.3, 0.4, 0.5],
    'feature2': [0.5, 0.4, 0.3, 0.2, 0.1],
    'label':    [0,   0,   1,   1,   1  ]
}
"""

data = {
    'feature1': [
        0.2, 0.25, 0.3, 0.35,    # inner points (label 0)
        0.0, 0.6, 0.6, 0.0       # outer points (label 1)
    ],
    'feature2': [
        0.2, 0.35, 0.2, 0.35,    # inner points (label 0)
        0.0, 0.0, 0.6, 0.6       # outer points (label 1)
    ],
    'label': [
        0, 0, 0, 0,              # inner points
        1, 1, 1, 1               # outer points
    ]
}

df = pd.DataFrame(data)
X = df[['feature1', 'feature2']].values
y = df['label'].values

# Create a Neural Network
"""
model = Sequential()
model.add(Dense(8, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
"""
#Other way to create it
model = Sequential([
    Dense(20, input_dim=2, activation='relu'),
    Dense(15, activation='relu'),
    Dense(15, activation='relu'),
    Dense(1, activation='sigmoid')
])


# Compile the Model
model.compile(
    loss='binary_crossentropy',#'mse',#
    optimizer='adam',
    metrics=['accuracy']
)

# Train the Model
model.fit(X, y, epochs=1000, batch_size=1, verbose=0)

# Make Predictions
test_data = np.array([[0.2, 0.5]])
prediction = model.predict(test_data)
predicted_label = (prediction > 0.5).astype(int)
print(predicted_label)


#-------- Plot the results --------

#Create a grid of points to cover the plot area
x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))

#Predict across the entire grid
grid_points = np.c_[xx.ravel(), yy.ravel()]
probs = model.predict(grid_points)
Z = (probs > 0.5).astype(int)
Z = Z.reshape(xx.shape)

#Plotting
plt.figure(figsize=(8, 6))

# Draw the decision boundary (the background color)
plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')

# Plot the training data
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='k', s=100, label='Training Data')

# Plot the test point (the prediction)
plt.scatter(test_data[0][0], test_data[0][1], color='red', marker='X', s=200, label=f'Test Point (Pred: {predicted_label[0][0]})')

plt.title("ANN Decision Boundary")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

