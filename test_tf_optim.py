import numpy as np
import tensorflow as tf
from bokeh.plotting import figure, output_file, show
from bokeh.palettes import Set2
from bokeh.layouts import gridplot
from tqdm import tqdm
import pandas as pd 
from scipy import sparse


output_file("test.html")
x = np.load('./x.npy')
t = np.linalg.norm(x, ord=2)
La = np.load('./laplacian.npy')

sess = tf.InteractiveSession()

n = x.shape[0]
m = La.shape[0]
k = 100
lam = 0.002

X = tf.placeholder(tf.float32, shape=(n,m))
L = tf.placeholder(tf.float32, shape=(m,m))
l = tf.placeholder(tf.float32, shape=())
A = tf.Variable(np.random.rand(n,k), dtype=tf.float32)
B = tf.Variable(np.random.rand(k,m), dtype=tf.float32)
zero = tf.constant(0, dtype=tf.float32)
tz = tf.constant(t, dtype=tf.float32)

bnorm = tf.norm(B, ord=2)
anorm = tf.norm(A, ord=2)
recon = tf.matmul(A, B)
norm = tf.reduce_sum(tf.subtract(X, recon) ** 2)
lap = l * tf.trace(tf.matmul(tf.matmul(B, L), tf.transpose(B)))
loss = norm - lap #+ bnorm + anorm

recon = tf.matmul(A, B)
where = tf.not_equal(tf.reshape(X,[-1]), zero)
indices = tf.where(where)
X_nonzero = tf.gather(tf.reshape(X, [-1]), indices)
recon_nonzero = tf.gather(tf.reshape(recon, [-1]), indices)
sum_err = tf.reduce_mean(tf.square(tf.subtract(X_nonzero, recon_nonzero)))
opt = tf.train.AdamOptimizer(0.1).minimize(loss)
tf.global_variables_initializer().run()
step = 200
losses = []
err = []
norms = []
laps = []
anorms = []
bnorms = []
sum_errs = []
for i in tqdm(range(0, step)):
    y = sess.run([opt, loss, norm, lap, sum_err, A, B], feed_dict={X:x, L:La, l:lam})
    losses.append(y[1])
    norms.append(y[2])
    laps.append(y[3])
    sum_errs.append(y[4])

print(sum_err)
p1 = figure(plot_width=300, plot_height=300, title='Frobenius Norm')
p1.line(range(0, step), norms, color=Set2[6][0])

p2 = figure(plot_width=300, plot_height=300, title='Total Loss')
p2.line(range(0, step), losses, color=Set2[6][1])

p3 = figure(plot_width=300, plot_height=300, title='Laplacian Loss')
p3.line(range(0, step), laps, color=Set2[6][2])

p5 = figure(plot_width=300, plot_height=300, title='Non-zero elements loss')
p5.line(range(0, step), sum_errs, color=Set2[6][5])

p = gridplot([[p1, p2], [p3, p5]])
show(p)
print(np.matrix(x)[19, 9])
print(np.matmul(y[5], y[6])[19, 9])
np.save('A5000.npy', A.eval())
np.save('B5000.npy', B.eval())
