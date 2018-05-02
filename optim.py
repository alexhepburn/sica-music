import numpy as np 
from scipy import optimize

n = 100
k = 5
m = 200

x = np.random.rand(n, m)
As = np.zeros((n, k)).flatten()
Bs = np.zeros((k, m)).flatten()

def f(l):
    A, B = l
    A = np.reshape(A, (n, k))
    B = np.reshape(B, (k, m))
    return np.linalg.norm(x-np.matmul(A,B), ord=2)

def con1(l):
    A = np.reshape(l[0], (n, k))
    I = np.identity(k)
    return np.linalg.norm(I-A, ord=2)

def con2(l):
    B = np.reshape(l[1], (k, m))
    I = np.identity(m)
    return np.linalg.norm(I-B, ord=2)

res = optimize.fmin_cobyla(f, [As, Bs], [con1, con2])
print(res)