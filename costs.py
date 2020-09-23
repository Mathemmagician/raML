
from abc import ABC, abstractmethod
import numpy as np

class Cost(ABC):
    'This is an abstract class for Cost functions'
    def __init__(self, name="Default Cost"):
        self.name = name
    
    @abstractmethod
    def calculate(self, Y, yhat):
        '''Y is data, yhat is prediction'''
        self.cost = None

    @abstractmethod
    def derivative(self, Y, yhat):
        self.dA = None
    
    def __str__(self):
        return f'{self.__class__.__name__} cost is {self.cost}'


class MSE(Cost):
    'Mean Squared Error Cost'
    def __init__(self):
        super().__init__("MSE")
    
    def calculate(self, Y, Yhat):
        '''Returns normalized J = sum(|| Yhat - Y || ^ 2) / 2m'''
        assert (Y.shape == Yhat.shape) and (Y.shape[0] == 1)

        self.cost = np.square(Y - Yhat).mean()
        return self.cost
    
    def derivative(self, Y, Yhat):
        '''Returns J' = Yhat - Y'''
        self.dA = Yhat - Y
        return self.dA
    
class CrossEntropy(Cost):
    '''Loss function for binary classification'''
    def __init__(self):
        super().__init__(name="CrossEntropy")
    
    def calculate(self, Y, Yhat):
        '''Math:
            -Y * log(Yhat) - (1 - Y) * log(1 - Yhat)
        '''
        self.cost = ((0 - Y) * np.log(Yhat) - (1 - Y) * np.log(1 - Yhat)).mean()
        return self.cost
    
    def derivative(self, Y, Yhat):
        assert Y.shape == Yhat.shape

        self.dA = (0 - Y) / Yhat + (1 - Y) / (1 - Yhat)
        return self.dA
