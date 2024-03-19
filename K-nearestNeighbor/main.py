from __future__ import print_function
import numpy as np
from time import time
from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
plt.rcParams['font.family'] = 'Dejavu Sans'

d, N = 1000, 10000
X = np.random.randn(N, d)
z = np.random.randn(d)

def dist_pp(z, x):
    d = z - x.reshape(z.shape)
    return np.sum(d*d)

def dist_ps_naive(z, X):
    N = X.shape[0]
    res = np.zeros((1, N))
    for i in range(N):
        res[0][i] = dist_pp(z, X[i])
    return res

def dist_ps_fast(z, X):
    X2 = np.sum(X*X, 1)
    z2 = np.sum(z*z)
    return X2 * z2 - 2*X.dot(z)

def myweight(distances):
    sigma2 = .4
    return np.exp(-distances**2/sigma2)

iris =datasets.load_iris()
iris_X = iris.data
iris_y = iris.target

np.random.seed(7)

X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=130)

model = neighbors.KNeighborsClassifier(n_neighbors= 7, p = 2, weights= myweight)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accurace of 7NN (customized weights): %.2f %%"%(100*accuracy_score(y_test, y_pred)))
 