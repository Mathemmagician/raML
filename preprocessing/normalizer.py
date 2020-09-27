
import numpy as np

class Normalizer():
    def __init__(self):
        pass

    def fit(self, X, bias=True, nans=False):
        '''
        Params
        ------
            X : numpy.ndarray
                2D array of size [features x sampels]
            bias : bool
                Ignore the bias row
            nans : bool
                If true, compute stdev ignoring Nans
        '''
        assert X.ndim == 2

        self.bias = bias
        self.nans = nans
        self.IN = X.shape[0]
        if self.nans:
            self.mu = X.nanmean(axis=1)
            self.stdev = X.nanstd(axis=1)
        else:
            self.mu = X.mean(axis=1)
            self.stdev = X.std(axis=1)

        self.stdev[self.stdev == 0] = 1 # cheating to avoid 0 st.dev.
        if bias: # cheating to avoid changing the bias column
            self.mu[0] = 0
        return self.apply(X)


    def apply(self, X):
        '''If stdev is 0, bad things happen. Needs fix. *maybe fixed*'''
        assert X.ndim == 2 and X.shape[0] == self.IN

        return (X - self.mu[:,None]) / self.stdev[:,None]

