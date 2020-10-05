
import pandas as pd
import numpy as np
from ..utils import format_data, cache


@cache
def Mnist(onehot=True):
    '''Famous digits dataset'''
    
    url = "https://www.python-course.eu/data/mnist/mnist_train.csv"
    df = pd.read_csv(url, header=None)

    X = df.iloc[:,1:].transpose().values
    Y = df.iloc[:,0].values

    n = Y.shape[0]

    if onehot:
        Z = np.zeros((10, n))
        Z[Y, np.arange(n)] = 1
        Y = Z

    return format_data(X, Y, n = 60000, f = 784, out=10)
    
    
