
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
