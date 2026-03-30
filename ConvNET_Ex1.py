import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Define kernels
# ---------------------------------------------------------
gaussian = tf.constant([
    [1, 2, 4, 2, 1],
    [2, 4, 8, 4, 2],
    [4, 8, 16, 8, 4],
    [2, 4, 8, 4, 2],
    [1, 2, 4, 2, 1] 
    ], tf.float32) / 256.0

laplacian = tf.constant([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
], tf.float32)

sobel_x = tf.constant([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], tf.float32)

sobel_y = tf.constant([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
], tf.float32)

gaussian_k = tf.reshape(gaussian, [5, 5, 1, 1])
laplacian_k = tf.reshape(laplacian, [3, 3, 1, 1])
sobel_x_k = tf.reshape(sobel_x, [3, 3, 1, 1])
sobel_y_k = tf.reshape(sobel_y, [3, 3, 1, 1])

# ---------------------------------------------------------
# 2. Load and preprocess image
# ---------------------------------------------------------
plt.rc('figure', autolayout=True)

image = tf.io.read_file("tiger.png") 
image = tf.io.decode_jpeg(image, channels=1)
image = tf.image.resize(image, (400, 400))
image = tf.image.convert_image_dtype(image, tf.float32)
image_batch = tf.expand_dims(image, 0)

# ---------------------------------------------------------
# 3. CNN Pipeline
# ---------------------------------------------------------
blurred = tf.nn.conv2d(image_batch, gaussian_k, strides=1, padding="SAME")
conv_output = tf.nn.conv2d(blurred, sobel_y_k, strides=1, padding="SAME")
relu_output = tf.nn.relu(conv_output)
n = 2
pool_output = tf.nn.pool(relu_output, window_shape=(n, n), pooling_type="MAX", strides=(n, n), padding="SAME")

# ---------------------------------------------------------
# 4. Visualization helper
# ---------------------------------------------------------
def show(img, title, mode='sequential'):
    img_data = tf.squeeze(img).numpy()
    
    if mode == 'diverging':
        # Force 0 to be exactly WHITE
        # This shows Red (+) and Blue (-) gradients
        limit = np.max(np.abs(img_data))
        plt.imshow(img_data, cmap='RdBu', vmin=-limit, vmax=limit)
    elif mode == 'grayscale':
        plt.imshow(img_data, cmap='gray')
    else:
        # Values are 0 to +Max
        plt.imshow(img_data, cmap='magma')
        
    plt.title(title, fontsize=10, fontweight='bold')
    plt.axis("off")

# ---------------------------------------------------------
# 5. Show Transformation
# ---------------------------------------------------------
plt.figure(figsize=(16, 7))

plt.subplot(1, 4, 1)
show(image_batch, "1. Original Image\n(Input Data)", mode='grayscale')

plt.subplot(1, 4, 2)
# Diverging mode shows the double edge effect of the Laplacian
show(conv_output, "2. Laplacian Convolution\n(Red:+, Blue:-)", mode='diverging')

plt.subplot(1, 4, 3)
# ReLU prunes the negative values, leaving only the positive edges
show(relu_output, "3. ReLU Activation\n(Pruning Negatives)")

plt.subplot(1, 4, 4)
# Pooling summarizes the remaining features
show(pool_output, "4. Max Pooling\n(Condensed Features)")

plt.show()
