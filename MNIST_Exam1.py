
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

# 1) Load data
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# 2) Scale to [0,1]
train_images = train_images.astype('float32') / 255.0
test_images  = test_images.astype('float32') / 255.0

# 3) Build a fully-connected (dense) network
#    NOTE: input_shape is (28, 28) because images are 2D without an explicit channel dim.
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# 4) Compile with sparse labels (integers 0..9)
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 5) Train 
model.fit(
    train_images,
    train_labels,
    epochs=20,
    verbose=0
)

# 6) Quick dataset info
print('Number of images in the training dataset:', train_images.shape[0])
print('Number of images in the testing dataset:',  test_images.shape[0])
print(f"Shape of one training image: {train_images[0].shape}")

# 7) Visualize a few training examples
fig, axes = plt.subplots(1, 10, figsize=(12, 2))
for i in range(10):
    axes[i].imshow(train_images[i], cmap='gray')
    axes[i].set_title(str(train_labels[i]))
    axes[i].axis('off')
plt.tight_layout()
plt.show()

# 8) Evaluate on test set
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
print(f'\nTest accuracy: {test_acc:.4f}')



