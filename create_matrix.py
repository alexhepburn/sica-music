import pandas as pd
import numpy as np 

# Take num_users of the most active users in the dataset
num_users = 1000
df = pd.read_csv('./train_triplets.txt', sep='\t', names=['user', 'track', 'plays'])
print(len(list(df.track.unique())))
users = df.groupby('user').sum().sort_values('plays', ascending=False).head(num_users).reset_index().user
df = df.loc[df['user'].isin(users)]
piv = pd.pivot_table(df, index='user', columns='track', values='plays', fill_value=0).values
#piv = df.pivot('user', 'track', 'plays').fillna(0).values
print(piv.shape)
np.save('matrix.npy', piv)