import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt 
from tqdm import tqdm

sess = tf.InteractiveSession()

n = 5
m = 5
k = 2
lam = 0.8

X = tf.placeholder(tf.float32, shape=(n,m))
L = tf.placeholder(tf.float32, shape=(m,m))
l = tf.placeholder(tf.float32, shape=())
A = tf.Variable(np.random.rand(n,k), dtype=tf.float32)
B = tf.Variable(np.random.rand(k,m), dtype=tf.float32)
loss = tf.square(tf.norm((X-tf.matmul(A, B)), ord='fro', axis=[0, 1])) - l * tf.trace(tf.matmul(tf.matmul(B, L), tf.transpose(B))) + 2*tf.norm(B, ord=2)

x = [[0, 0, 1, 1, 1],
    [1, 1, 0, 0, 1],
    [0, 1, 0, 1, 0],
    [1, 0, 1, 0, 1],
    [0, 1, 1, 1, 0]]

La = [[1, -1, 0, 0, 0],
     [-1, 2, 0, -1, 0],
     [0, 0, 2, -1, -1],
     [0, -1, -1, 3, -1],
     [0, 0, -1, -1, 2]]

#x = np.asarray(x)
#L = np.asarray(L)

opt = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
tf.global_variables_initializer().run()
step = 1000
losses = []
for i in tqdm(range(0, step)):
    y = sess.run([opt, loss], feed_dict={X:x, L:La, l:lam})
    losses.append(y[1])
plt.plot(range(0, step), losses)
plt.show()
print(np.matrix(x))
print(tf.matmul(A,B).eval())
np.save('A.npy', A.eval())
np.save('B.npy', B.eval())
