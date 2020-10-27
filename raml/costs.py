
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
        assert (Y.shape == Yhat.shape)

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


class CategoricalCrossEntropy(Cost):
    '''Loss function for multi-class classification'''
    def __init__(self):
        super().__init__(name="CategoricalCrossEntropy")
    
    def calculate(self, Y, Yhat):
        '''Math:
            -summation{yi ln( yhat i)}
        '''
        assert (Y.shape == Yhat.shape)

        self.cost = -np.sum( Y * np.log(Yhat) , axis=0 ).mean()
        return self.cost
    
    def derivative(self, Y, Yhat):
        '''Note: The return of this function is NOT a derivative. 
        It is assumed that the derivative is going to be fed into Softmax layer
        during the propagation. Since derivative of the softmax layer
        dZ = Yhat - Y, it's just easier to return that.
        '''
        assert Y.shape == Yhat.shape

        self.dA = Yhat - Y
        return self.dA