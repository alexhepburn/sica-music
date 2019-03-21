#%%
import scipy 
import numpy as np 
from scipy import optimize
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA
import networkx as nx 

n=1000

#X1 = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], int(n/2))
#X2 = np.random.multivariate_normal([10, 0], [[0.5, 0], [0, 0.5]], int(n/2))
X1 = np.random.normal(loc=0, scale=1, size=(int(n/2), 1))
X2 = np.random.normal(loc=10, scale=1, size=(int(n/2), 1))
X3 = np.random.choice([-1, 1], size=(n, 1), p=[0.5, 0.5])
X4 = np.random.normal(loc=0, scale=1, size=(n, 1))
X5 = np.random.normal(loc=0, scale=1, size=(n, 1))
X6 = np.random.normal(loc=0, scale=1, size=(n, 1))
X6 = np.random.normal(loc=0, scale=1, size=(n, 1))
X7 = np.random.normal(loc=0, scale=1, size=(n, 1))
X8 = np.random.normal(loc=0, scale=1, size=(n, 1))
X9 = np.random.normal(loc=0, scale=1, size=(n, 1))
X10 = np.random.normal(loc=0, scale=1, size=(n, 1))

X = np.concatenate([X1,X2])
X = np.hstack([X, X3, X4, X5, X6, X7, X8, X9, X10])
dist = np.absolute(distance_matrix(X, X))
X = np.hstack([X, X3])

# estimate b & c
b = np.mean(dist)
c = np.mean(np.linalg.norm(X, ord=2, axis=1))

e = 8
ind = dist<e 
ind2 = dist>=e 
dist[ind] = 1
dist[ind2] = 0
np.fill_diagonal(dist, 0)

G = nx.from_numpy_matrix(dist)
nx.draw_networkx(G)
pca = PCA(2)
X_pca = pca.fit_transform(X)

d = X.shape[1]
num_edges = G.number_of_edges()
L = nx.laplacian_matrix(G).todense()
I = np.identity(n)
eig, vec = np.linalg.eig(L)
eig = np.absolute(np.real(eig).flatten())

def lagrange(l):
    l1, l2 = l
    t = np.sum(np.log((2*l1*eig)/num_edges + (2*l2)/n))
    return (-d/2)*t + ((n*d)/2)*np.log(2*np.pi) + l1*b + l2*c

res = optimize.fmin(lagrange, [10, 10])
temp = (res[0]/num_edges) * L + ((res[1]/n) * I)
eig, W = np.linalg.eig(X.T * temp * X)
print(X.shape)
print(temp.shape)
sys.exit(0)
print(res)
print(num_edges)
print(pca.components_[0, :])
print(W[:, 0])
transformed = np.matmul(X, W).view(type=np.ndarray)
f, ((ax1), (ax2)) = plt.subplots(2, 1, sharex='col', sharey='row')
plotx = np.random.normal(loc=0, scale=1, size=(n,1))
ax1.scatter(X[:, 0], plotx, c=transformed[:, 0])
ax1.set_title('First SICA component')
#ax2.scatter(X[:, 0], plotx, c=transformed[:, 1])
#ax2.set_title('Second SICA component')
ax2.scatter(X[:, 0], plotx, c=X_pca[:,0])
ax2.set_title('First PCA component')
#ax4.scatter(X[:, 0], plotx, c=X_pca[:,1])
#ax4.set_title('Second PCA component')
plt.show()