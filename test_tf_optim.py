import numpy as np
import tensorflow as tf
from bokeh.plotting import figure, output_file, show
from bokeh.palettes import Set2
from bokeh.layouts import gridplot
from tqdm import tqdm
import pandas as pd 
from scipy import sparse
import random
import pickle
import matplotlib.pyplot as plt 

df = pd.read_hdf('./listens_with_genres.h5')
piv = pd.pivot_table(df, index='user', columns='song', values='plays', fill_value=0)
with open('./genre_dict.pickle', 'rb') as f:
    genre_dict = pickle.load(f)
genres = [genre_dict[x] for x in list(piv.columns)]
output_file("regular.html")
x = np.load('./x.npy')
songcount = np.zeros((x.shape[0],), dtype=int)
songs = []
for i in range(0,x.shape[0]):
    nonzero = np.nonzero(x[i, :])[0]
    songcount[i] = nonzero.shape[0]
    songs.append(nonzero)

topten = np.argpartition(songcount, -10)[-10:]
highest_true = x[topten[-1], :]
check_ind = []
correct_check_ind = []
for i in range(0, topten.shape[0]):
    take_ind = np.random.choice(songs[topten[i]], 5)
    check_ind.append((topten[i], take_ind))
    correct_check_ind.append(x[topten[i], take_ind])
    x[topten[i], take_ind] = 0
s = (x!=0).astype(int)

t = np.linalg.norm(x, ord=2)
La = np.load('./laplacian.npy')

sess = tf.InteractiveSession()

n = x.shape[0]
m = La.shape[0]
k = 10
lam = 0.00


save_scores = []
lambdas = []
for j in tqdm(range(0, 20)):
    X = tf.placeholder(tf.float32, shape=(n,m))
    S = tf.placeholder(tf.float32, shape=(n,m))
    L = tf.placeholder(tf.float32, shape=(m,m))
    l = tf.placeholder(tf.float32, shape=())
    zero = tf.constant(0, dtype=tf.float32)
    tz = tf.constant(t, dtype=tf.float32)
    A = tf.Variable(np.random.rand(n,k), dtype=tf.float32)
    B = tf.Variable(np.random.rand(k,m), dtype=tf.float32)
    bnorm = tf.square(tf.norm(B, ord='fro', axis=(0, 1)))
    anorm = tf.square(tf.norm(A, ord='fro', axis=(0, 1)))
    recon = tf.matmul(A, B)
    norm = tf.square(tf.norm(tf.multiply(tf.subtract(X, recon), S), ord='fro', axis=(0, 1))) + 0.05 *  (bnorm + anorm)
    #norm2 = tf.reduce_sum(tf.multiply(tf.subtract(X, recon), S) ** 2)
    lap = l * tf.trace(tf.matmul(tf.matmul(B, L), tf.transpose(B)))
    loss = norm - lap

    #X_nonzero = tf.gather(tf.reshape(X, [-1]), indices)
    #recon_nonzero = tf.gather(tf.reshape(recon, [-1]), indices)
    #sum_err = tf.reduce_mean(tf.square(tf.subtract(X_nonzero, recon_nonzero)))
    opt = tf.train.AdamOptimizer(0.1).minimize(loss)
    tf.global_variables_initializer().run()
    step = 1000

    losses = []
    err = []
    norms = []
    laps = []
    anorms = []
    bnorms = []
    sum_errs = []
    scores = []
    for i in range(0, step):
        y = sess.run([opt, loss, norm, lap, A, B], feed_dict={X:x, L:La, l:lam, S:s})
        #losses.append(y[1])
        #norms.append(y[2])
        #laps.append(y[3])
        temp = []
        construct = np.matmul(y[4], y[5])
        for i in range(0, len(check_ind)):
            temp.append(np.sqrt((correct_check_ind[i] - construct[check_ind[i][0], check_ind[i][1]])**2).sum())
        scores.append(np.sum(temp))
    save_scores.append(scores[-1:])
    lambdas.append(lam)
    lam = lam + 0.1
np.save('./scores.npy', save_scores)
np.save('./lambdas.npy', lambdas)
plt.plot(lambdas, save_scores)
plt.show()
sys.exit(0)

print('Top ten ratings: ', np.argsort(-highest_true)[0:10])
highest_con = construct[topten[-1], :]
sorthigh = np.argsort(-highest_con)
print('Top ten ratings in recon: ', sorthigh[0:10])
highest_con[np.nonzero(x[topten[-1], :])] = 0
print('Recommended: ', np.argsort(-highest_con)[0:10])
p1 = figure(plot_width=300, plot_height=300, title='Frobenius Norm')
p1.line(range(0, step), norms, color=Set2[6][0])

p2 = figure(plot_width=300, plot_height=300, title='Total Loss')
p2.line(range(0, step), losses, color=Set2[6][1])

p3 = figure(plot_width=300, plot_height=300, title='Laplacian Loss')
p3.line(range(0, step), laps, color=Set2[6][2])

p4 = figure(plot_width=300, plot_height=300, title='RMSE missing values reconstruction')
p4.line(range(0, step), scores, color=Set2[6][3])

p = gridplot([[p1, p2], [p3, p4]])
show(p)
scores = []
np.save('A5000.npy', A.eval())
np.save('B5000.npy', B.eval())