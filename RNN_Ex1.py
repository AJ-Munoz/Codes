import numpy as np
import tensorflow as tf
import os

# 1. Dataset
quotes = "I'll be back. Hasta la vista, baby. Come with me if you want to live "
chars = sorted(list(set(quotes)))
char_to_ix = {c: i for i, c in enumerate(chars)}
ix_to_char = {i: c for i, c in enumerate(chars)}

# 2. Memory Prep and Encoding
seq_len = 5
X, y = [], []
for i in range(len(quotes) - seq_len):
    X.append([char_to_ix[c] for c in quotes[i:i+seq_len]])
    y.append(char_to_ix[quotes[i+seq_len]])

X_oh = tf.one_hot(X, len(chars))
y_oh = tf.one_hot(y, len(chars))

# 3. Model Architecture and Training: Teaching the Terminator to Speak
model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(64, input_shape=(seq_len, len(chars))),
    #tf.keras.layers.LSTM(64, input_shape=(seq_len, len(chars))),
    #tf.keras.layers.GRU(64, input_shape=(seq_len, len(chars))),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(len(chars), activation='softmax')
])
model.compile(loss='categorical_crossentropy', optimizer='adam')

os.system('cls' if os.name == 'nt' else 'clear')
print("Terminator is learning to speak... 🤖😎")
model.fit(X_oh, y_oh, epochs=200, verbose=0)

# 4. Text Generation
input_str = "I'll "
generated = input_str

for _ in range(10):
    x_input = np.array([[char_to_ix[c] for c in generated[-seq_len:]]])
    x_oh = tf.one_hot(x_input, len(chars))
    pred = model.predict(x_oh, verbose=0)
    next_char = ix_to_char[np.argmax(pred)]
    generated += next_char

print(f"RNN says: \"{generated}\"")
