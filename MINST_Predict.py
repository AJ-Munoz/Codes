import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os


# 9) Load and preprocess a custom image
from PIL import Image
path = "my_digit.png"            
img = Image.open(path).convert('L')           # grayscale
img_28 = img.resize((28, 28), Image.LANCZOS)
x = np.array(img_28).astype('float32') / 255.0      # shape: (28, 28)
x = np.expand_dims(x, axis=0)                       # shape: (1, 28, 28)

# 10) Predict with Fully Connected Model
# Trained with 100 epochs, shuffle=True and dropout layers for better generalization.
new_model = models.load_model("my_model.keras")
prediction = new_model.predict(x, verbose=0)
pred_class = int(np.argmax(prediction))

# 11) Predict with CNN Model
# Trained with only 20 epochs 
cnn_model = models.load_model("my_cnn_model.keras")
# CNN needs 4 dims: (1, 28, 28, 1)
x_cnn = np.expand_dims(x, axis=-1)  
cnn_prediction = cnn_model.predict(x_cnn, verbose=0)
cnn_pred_class = int(np.argmax(cnn_prediction))

os.system("cls" if os.name == "nt" else "clear")
print("Predicted digit with Fully Connected Model:", pred_class)
print("Probabilities:", np.round(prediction[0], 3))
print("Predicted digit with CNN Model:", cnn_pred_class)
print("Probabilities:", np.round(cnn_prediction[0], 3))