import numpy as np

def relu(x, size, y):
  # print(f"x.shape:{x.shape}")
  # print(f"size:{size}")
  for i in range(size): #range()でいいか
    #y[i] = std::max(x[i], .0f);
    y[i] = np.maximum(0, x[i])
  return y