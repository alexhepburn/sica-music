import numpy as np
import csv 

X = np.load('./matrix/X.npy')
to_write = []
for i in range(X.shape[0]):
    row = X[i, :]
    ind = np.where(row!=0)[0]
    for song in ind:
        rating = X[i, song]
        if rating > 5:
            rating = 5
        to_write.append([i+1, song+1, rating])

np.savetxt('./matrix/surprise_csv.csv', to_write, delimiter=',', fmt='%s')
