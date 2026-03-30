import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# 1) Load data
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# 2) Scale and Reshape for CNN
# CNNs expect (batch, height, width, channels). MNIST is grayscale, so channels = 1.
train_images = train_images.reshape(-1, 28, 28, 1).astype('float32') / 255.0
test_images  = test_images.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# 3) Build CNN Architecture
model = models.Sequential([
    # Convolutional layers extract features
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Flatten to transition to Dense layers
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax'),
])

model.summary()

# 4) Compile
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 5) Train (Epochs reduced to 20; CNNs converge much faster than Dense nets)
model.fit(train_images, train_labels, epochs=20, shuffle=True, verbose=1)

# 6) Info and Visualization
print(f"Shape of one training image: {train_images[0].shape}")

# 7) Evaluate and Save
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
print(f'\nTest accuracy: {test_acc:.4f}')
model.save("my_cnn_model.keras")

"""
# 8) Load and preprocess a custom image
path = "my_digit.png"
try:
    img = Image.open(path).convert('L')
    img_28 = img.resize((28, 28), Image.Resampling.LANCZOS)
    x = np.array(img_28).astype('float32') / 255.0
    
    # CNN needs 4 dims: (1, 28, 28, 1)
    x = np.expand_dims(x, axis=(0, -1)) 

    # 10) Predict
    prediction = model.predict(x, verbose=0)
    pred_class = int(np.argmax(prediction))
    print("Predicted digit:", pred_class)
    print("Probabilities:", np.round(prediction[0], 3))
except:
    print("Image not found at path.")
"""
