from surprise import SVD, SVDpp, Dataset, Reader
from surprise.model_selection import cross_validate
from surprise.model_selection import KFold
from surprise import accuracy

import numpy as np
import pandas as pd 

file_path = './matrix/surprise_csv.csv'
reader = Reader(line_format='user item rating', sep=',', rating_range=(0, 5))
data = Dataset.load_from_file(file_path, reader=reader)
algo = SVD()
algo2 = SVDpp()

kf = KFold(n_splits=3)
for trainset, testset in kf.split(data):
    algo.fit(trainset)
    predictions = algo.test(testset)
    print(predictions[0])
    accuracy.rmse(predictions, verbose=True)

for trainset, testset in kf.split(data):
    algo2.fit(trainset)
    predictions = algo.test(testset)
    print(predictions[0])
    accuracy.rmse(predictions, verbose=True)
