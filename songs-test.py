import numpy as np

x = [[0, 0, 1, 1, 1],
    [1, 1, 0, 0, 1],
    [0, 1, 0, 1, 0],
    [1, 0, 1, 0, 1],
    [0, 1, 1, 1, 0]]

L = [[1, -1, 0, 0, 0],
     [-1, 2, 0, -1, 0],
     [0, 0, 2, -1, -1],
     [0, -1, -1, 3, -1],
     [0, 0, -1, -1, 2]]

x = np.asarray(x)
L = np.asarray(L)

I = np.identity(5)
lam = 0.8
temp = (I - lam * L)
S = np.matmul(np.linalg.inv(temp), x)
print(S)