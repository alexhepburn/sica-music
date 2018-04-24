#%%
import scipy 
import numpy as np 
from scipy import optimize
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA
import networkx as nx 

n=25

X = np.random.multivariate_normal([0, 0], [[5, 10], [3, 10]], n)
dist = np.absolute(distance_matrix(X, X))
e = 10
ind = dist<e 
ind2 = dist>=e 
dist[ind] = 1
dist[ind2] = 0
np.fill_diagonal(dist, 0)

G = nx.from_numpy_matrix(dist)

pca = PCA(1)
X_pca = pca.fit_transform(X)

plt.scatter(X[:, 0], X[:, 1], c=X_pca[:, 0])
#plt.show()

b = 1
c = 1
d = 2
num_edges = G.number_of_edges()
L = nx.laplacian_matrix(G).todense()
I = np.identity(n)
eig, vec = np.linalg.eig(L)
eig = np.real(eig).flatten()

def lagrange(l):
    l1, l2 = l
    t = np.sum(np.log((2*l1*eig)/num_edges + (2*l2)/n))
    return (-d/2)*t + ((n*d)/2)*np.log(2*np.pi) + l1*b + l2*c

res = optimize.fmin_cg(lagrange, [1, 1])
temp = (res[0]/num_edges) * L + (res[1]/n) * I
eig, W = np.linalg.eig(X.T * temp * X)
transformed = X * W
plt.figure()
col = np.squeeze(np.asarray(transformed[:, 0]))
print(res)
plt.scatter(X[:, 0], X[:, 1], c=col)
plt.figure()
plt.scatter([transformed[:, 0]], [transformed[:, 1]])
