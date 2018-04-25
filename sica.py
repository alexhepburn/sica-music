#%%
import scipy 
import numpy as np 
from scipy import optimize
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA
import networkx as nx 

n=400

X1 = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], int(n/2))
X2 = np.random.multivariate_normal([10, 0], [[1, 0], [0, 1]], int(n/2))
X3 = np.random.choice([-1, 1], size=(n, 1), p=[0.5, 0.5])
#X4 = np.random.normal(loc=0, scale=1, size=(n, 1))
#X5 = np.random.normal(loc=0, scale=1, size=(n, 1))
#X6 = np.random.normal(loc=0, scale=1, size=(n, 1))

X = np.concatenate([X1,X2])
dist = np.absolute(distance_matrix(X, X))
X = np.hstack([X, X3])
#X = np.hstack([X, X4, X5, X6])

# estimate b & c
b = np.mean(dist)
c = np.mean(np.linalg.norm(X, ord=2, axis=1))

e = 1
ind = dist<e 
ind2 = dist>=e 
dist[ind] = 1
dist[ind2] = 0
np.fill_diagonal(dist, 0)

G = nx.from_numpy_matrix(dist)
nx.draw_networkx(G)
plt.show()
pca = PCA(2)
X_pca = pca.fit_transform(X)

plt.scatter(X[:, 0], X[:, 1], c=X_pca[:, 0])
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=X_pca[:, 1])

d = 3
num_edges = G.number_of_edges()
L = nx.laplacian_matrix(G).todense()
I = np.identity(n)
eig, vec = np.linalg.eig(L)
eig = np.absolute(np.real(eig).flatten())
def lagrange(l):
    l1, l2 = l
    t = np.sum(np.log((2*l1*eig)/num_edges + (2*l2)/n))
    return (-d/2)*t + ((n*d)/2)*np.log(2*np.pi) + l1*b + l2*c

res = optimize.fmin_cg(lagrange, [1, 1])
temp = (res[0]/num_edges) * L + ((res[1]/n) * I)
eig, W = np.linalg.eig(X.T * temp * X)
print(pca.components_[0, :])
print(W[:, 0])
transformed = np.matmul(X, W).view(type=np.ndarray)
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=transformed[:, 0])
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=transformed[:, 1])
plt.figure()
plt.scatter(transformed[:, 0], transformed[:, 1])
plt.figure()
plt.scatter(X_pca[:, 0], X_pca[:, 1])