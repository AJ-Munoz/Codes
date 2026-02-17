import os
# Hides INFO and WARNING logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np

# 9) Load and preprocess a custom image
from PIL import Image
path = "my_digit.png"            
img = Image.open(path).convert('L')           # grayscale
img_28 = img.resize((28, 28), Image.LANCZOS)
x = np.array(img_28).astype('float32') / 255.0      # shape: (28, 28)
x = np.expand_dims(x, axis=0)                       # shape: (1, 28, 28)

# 10) Predict
new_model = tf.keras.models.load_model("my_model.h5")
prediction = new_model.predict(x, verbose=0)
pred_class = int(np.argmax(prediction))
print("Predicted digit:", pred_class)
print("Probabilities:", np.round(prediction[0], 3))