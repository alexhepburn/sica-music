import numpy as np
import tensorflow as tf

sess = tf.InteractiveSession()

n = 5
m = 5
k = 2

X = tf.placeholder(tf.float32, shape=(n,m))
L = tf.placeholder(tf.float32, shape=(m,m))
l = tf.placeholder(tf.float32, shape=())
A = tf.Variable(np.random.rand(n,k), dtype=tf.float32)
B = tf.Variable(np.random.rand(k,m), dtype=tf.float32)

mat = tf.square(tf.norm((X-tf.matmul(A,B))))
mat = tf.matmul(tf.matmul(tf.transpose(B), L), B)
loss = tf.square(tf.norm((X-tf.matmul(A, B)), ord='euclidean')) - l * tf.matmul(tf.matmul(B, L), tf.transpose(B))

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

opt = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
tf.global_variables_initializer().run()
y = sess.run([opt, loss, mat], feed_dict={X:x, L:La, l:0.5})
print(y[2])
