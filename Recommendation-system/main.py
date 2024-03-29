from __future__ import print_function
import numpy as np
import pandas as pd

u_cols = ['user', 'age', 'sex, occupation', 'zip_code']
users = pd.read_csv('Recommender-system/Data/ml-100k/u.user', sep='|', names=u_cols, encoding='latin-1')
n_users = users.shape[0]
print('Number of users:', n_users)

r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings_base = pd.read_csv('Recommender-system/Data/ml-100k/ub.base', sep='\t', names=r_cols, encoding='latin-1')
ratings_test = pd.read_csv('Recommender-system/Data/ml-100k/ub.test', sep='\t', names=r_cols, encoding='latin-1')

rate_train = ratings_base.values
rate_test = ratings_test.values

print('Number of traing rates:', rate_train.shape[0])
print('Number of test rates:', rate_test.shape[0])