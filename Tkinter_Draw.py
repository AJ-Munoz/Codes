import tkinter as tk
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

SIZE = 280   # Large canvas for smooth drawing5
BRUSH = 10   # Brush diameter (pixels)

# Backing image we will convert to 28x28
img = Image.new('L', (SIZE, SIZE), 0)      # black background
draw = ImageDraw.Draw(img)

def paint(event):
    x, y = event.x, event.y
    r = BRUSH // 2
    canvas.create_oval(x-r, y-r, x+r, y+r, fill='white', outline='white')
    draw.ellipse([x-r, y-r, x+r, y+r], fill=255)

root = tk.Tk()
root.title("Draw a digit (hold left mouse button and drag). Close to predict.")
canvas = tk.Canvas(root, width=SIZE, height=SIZE, bg='black')
canvas.pack()
canvas.bind("<B1-Motion>", paint)

btn = tk.Button(root, text="Close & Predict", command=root.destroy)
btn.pack()
root.mainloop()

# Save to disk
save_path = "my_digit.png"
img.save(save_path)
print(f"Saved image to: {save_path}")
