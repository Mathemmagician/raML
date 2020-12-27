
from abc import ABC, abstractmethod
import numpy as np


class Metric(ABC):
    def __init__(self, name='PlaceHolder'):
        self.name = name
        self.value = None

    @abstractmethod
    def calculate(self, Y, Yhat):
        pass

    def __str__(self):
        return f'{self.name} metric : {self.value}'


class MSE(Metric):
    def __init__(self):
        super().__init__(name='MSE')

    def calculate(self, Y, Yhat):
        self.value = np.square(Y - Yhat).mean() 
        return self.value


class RMSE(Metric):
    def __init__(self):
        super().__init__(name='RMSE')

    def calculate(self, Y, Yhat):
        self.value = np.sqrt(np.square(Y - Yhat).mean())
        return self.value


class CategoricalAccuracy(Metric):
    '''I've invented it. Made it exists, but be careful'''
    def __init__(self):
        super().__init__(name='CategoricalAccuracy')

    def calculate(self, Y, Yhat):

        self.value = (np.argmax(Y, axis=0) == np.argmax(Yhat, axis=0)).sum() / Y.shape[1]
        return self.value
    



