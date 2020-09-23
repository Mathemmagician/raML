
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
    'Applies iddentity'
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

