import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Create and Load Dataset
data = {
    'feature1': [0.1, 0.2, 0.3, 0.4, 0.5],
    'feature2': [0.5, 0.4, 0.3, 0.2, 0.1],
    'label':    [0,   0,   1,   1,   1]
}

df = pd.DataFrame(data)
X = df[['feature1', 'feature2']].values
y = df['label'].values
N = X.shape[0]


# Create a Neural Network
model = Sequential()
model.add(Dense(8, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))   # keep sigmoid for probabilities

# Compile the Model with QUADRATIC loss (Brier score)
model.compile(
    loss='mse',          # <-- quadratic loss on probability error
    optimizer='adam',
    metrics=['accuracy']
)

# Train the Model
model.fit(X, y, epochs=500, batch_size=N, verbose=0)

# Make Predictions
test_data = np.array([[0.2, 0.5]])
prediction = model.predict(test_data, verbose=0)   # probability in [0,1]
predicted_label = (prediction > 0.5).astype(int)
print("Predicted label (MSE):", predicted_label)

# -------- Plot the results --------
# Create a grid of points to cover the plot area
x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))

# Predict across the entire grid
grid_points = np.c_[xx.ravel(), yy.ravel()]
probs = model.predict(grid_points, verbose=0)
Z = (probs > 0.5).astype(int).reshape(xx.shape)

# Plotting
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='k', s=100, label='Training Data')
plt.scatter(test_data[0][0], test_data[0][1], color='red', marker='X', s=200,
            label=f'Test Point (Pred: {predicted_label[0][0]})')
plt.title("ANN Decision Boundary (MSE / Brier)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
