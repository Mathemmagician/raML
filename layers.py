
from abc import ABC, abstractmethod
import numpy as np

from activations import Identity

class Layer(ABC):
    def __init__(self, input_shape, output_shape=None):
        ''' IN  - number of neuron of input layer
            OUT - number of neurons of output layer
            n   - number of samples
        '''
        IN, n = input_shape
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.IN = IN
        self.OUT = output_shape[0]
        self.n = n
    
    @abstractmethod
    def forward(self, X):
        pass
    

class Dense(Layer):
    def __init__(self, input_shape, output_shape=(1,), activation=Identity):
        ''' W is the weights matrix | [in x out] 
            Z is Sum(w_i * x_i)     | [out x n]
            A is activation.apply(Z)| [out x n]
        '''
        super().__init__(input_shape, output_shape)
        self.activation = activation()

        self.W = np.random.rand(self.OUT, self.IN)
        self.X = None # Input
        self.Z = None # Weighted Sum
        self.A = None # Activation(Z)

        self.alpha = 0.0001
    
    def forward(self, X):
        '''Applies forward propagation to inputs X, i.e.

        Math:
            self.Z = W * X
            self.A = a(Z)
        '''
        assert X.shape == self.input_shape or X.shape == self.input_shape[:-1]

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
