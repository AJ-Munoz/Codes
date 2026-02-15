import numpy as np

x = np.array([1, 2])
y = x#.copy() #Necessary
y *= 2  # This modifies the data in-place!

print(x) # [2, 4] (Uh oh, x changed!)