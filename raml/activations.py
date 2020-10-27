
from abc import ABC, abstractmethod
import numpy as np

class Activation(ABC):
    'This is an abstract class for activation functions'
    def __init__(self, name):
        self.name = name
        
    @abstractmethod
    def apply(self, Z):
        return
    
    @abstractmethod
    def derivative(self, Z, **kwargs):
        return

class Identity(Activation):
    'Applies identity'
    def __init__(self, name='Identity'):
        super().__init__(name)
    
    def apply(self, Z):
        return Z
    
    def derivative(self, Z, **kwargs):
        return np.full(Z.shape, 1)

class Sigmoid(Activation):
    'Applies sigmoid'
    def __init__(self, name='Identity'):
        super().__init__(name)
    
    def apply(self, Z):
        return 1 / (1 + np.exp(-Z))
    
    def derivative(self, Z, A):
        return A * (1 - A)

class Softmax(Activation):
    'Applies softmax'
    def __init__(self, name='Softmax'):
        super().__init__(name)
    
    def apply(self, Z):
        ''' Normalizes to probability '''
        # e_Z = np.exp(Z - np.max(Z))
        # return e_Z / e_Z.sum(axis=0) 
        assert len(Z.shape) == 2
        s = np.max(Z, axis=0) #.reshape(1, -1)
        e_Z = np.exp(Z - s)
        div = np.sum(e_Z, axis=0) #.reshape(1, -1)
        return e_Z / div
    
    def derivative(self, Z, **kwargs):
        '''WARNING: The behavior of this function is abnormal due to the math of softmax
        
        Z ---softmax---> A ---CategoricalCrossEntropy---> Loss
        
        Turns out that dZ = Yhat - Y, therefore (Yhat - Y) is passed in backpropagation
        
        Input:
            dA = Yhat - Y
        '''

        return 1


class Relu(Activation):
    'Applies Rectified Linear Unit'
    def __init__(self, name='Relu'):
        super().__init__(name)
    
    def apply(self, Z):
        return Z.clip(min=0)
    
    def derivative(self, Z, **kwargs):
        return np.greater(Z, 0).astype(int)

class LeakyRelu(Activation):
    'Applies Leaky Relu'
    def __init__(self, name='Leaky Relu', leak=0.01):
        super().__init__(name)
        self.leak = 0.01
    
    def apply(self, Z):
        return np.where(Z > 0, Z, Z * self.leak)
    
    def derivative(self, Z, **kwargs):
        return np.where(Z > 0, 1, self.leak)



def test_softmax():
    soft = Softmax()

    x1 = np.array([[1, 2, 3, 6]]).T

    print(x1)

    print(soft.apply(x1))



if __name__ == '__main__':
    # Testing
    test_softmax()

