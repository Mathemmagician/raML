
from abc import ABC, abstractmethod
import numpy as np

from .activations import Identity

class Layer(ABC):
    def __init__(self, size, input_shape=(None, None)):
        '''
        Params
        ------
            size : int
                number of neurons (size) of output layer
            input_shape : (int, int)
                (number of input features, number of samples) *only required for first layer*
        '''
        self.size = size
        self.IN, self.n = self.input_shape = input_shape

    def compile(self, input_shape, output_shape):
        ''' 
        Notes
        -----
            IN  - number of neurons (size) of input layer
            OUT - number of neurons (size) of output layer
            n   - number of samples
        '''
        assert len(input_shape)== 2

        IN, n = input_shape
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.IN = IN
        self.OUT = output_shape[0]
        self.n = n
    
    @abstractmethod
    def forward(self, X):
        pass

    @abstractmethod
    def backward(self, dA):
        pass


class Dense(Layer):
    def __init__(self, size, input_shape=(None, None), activation=Identity):
        super().__init__(size, input_shape)
        self.activation = activation()

    def compile(self, input_shape, output_shape=(1,)):
        ''' W is the weights matrix | [in x out] 
            Z is Sum(w_i * x_i)     | [out x n]
            A is activation.apply(Z)| [out x n]
        '''
        super().compile(input_shape, output_shape)

        #self.W = np.random.rand(self.OUT, self.IN)
        self.W = np.random.randn(self.OUT, self.IN) * np.sqrt(2 / (self.IN + self.OUT))
        # Important note: for tanh: 1/self.IN, Relu: 2/self.IN. Instead, I'm using new theory
        self.alpha = 0.001 # Place holder for optimizer
    
    def forward(self, X):
        '''Applies forward propagation to inputs X, i.e.
            self.Z = W * X
            self.A = a(Z)
        '''
        assert X.ndim == 2 and X.shape[0] == self.input_shape[0]

        self.X = X
        self.Z = np.dot(self.W, self.X)
        self.A = self.activation.apply(self.Z)

        return self.A
    
    def backward(self, dA):
        '''Given derivatives of next layer, adjust the weights

        Math:
            dZ = dA .* a'(Z), .* - element wise multiplication
            dW = dZ dot X.T
            dX = dW.T dot dZ
        
        Params:
            dA := partial derivative dJ / dA

        Notes:    
            dX is dA of left layer
        '''
        assert dA.shape == self.Z.shape

        dZ = dA * self.activation.derivative(self.Z, A=self.A)
        assert dZ.shape == dA.shape == (self.OUT, self.n)

        dW = np.dot(dZ, self.X.transpose()) / self.n
        assert dW.shape == self.W.shape == (self.OUT, self.IN)

        dX = np.dot(self.W.transpose(), dZ)

        self.W = self.W - self.alpha * dW

        return dX, dW


class Lambda(Layer):
    '''Gotta think about this one'''
    def __init__(self, function):
        self.function = function

    def compile(self):
        pass

    def forward(self, X):
        return

    def backward(self, dA):
        pass