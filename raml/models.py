
from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np
from tqdm import tqdm
from .utils import raml_tqdm

class Model:
    def __init__(self):
        pass

    def compile(self, cost, optimizer = None, metrics=[]):
        self.cost = cost
        self.optimizer = optimizer
        self.metrics = metrics

    def fit(self, X, Y):
        pass

    def evaluate(self, Y, yhat):
        pass


class Sequential(Model):
    def __init__(self, layers=[]):
        self.layers = layers
    
    def add(self, layer):
        self.layers.append(layer)
    
    def compile(self, *args, **kwargs):
        '''Iteratively compiles every layer'''
        super().compile(*args, **kwargs)

        n = self.layers[0].n
        input_size = self.layers[0].IN
        assert n and input_size

        for layer in self.layers:
            layer.compile(input_shape=(input_size,n), output_shape=(layer.size, n))
            input_size = layer.size
    

    def fit(self, X, Y, epochs, epochstep):
        '''
        Params:
            X : np.ndarray  | input data
            Y : np.ndarray  | output data
            epochs : int    | number of training steps
            epochstep : int | how often the progress will be displayed
        '''

        self.history = {each : [] for each in ["Loss"] + [metric.name for metric in self.metrics]}

        for i in (pbar := raml_tqdm(range(epochs))):

            # Forward Propagation
            Yhat = self.forward(X)

            # Computing and recording the loss
            self.history["Loss"].append(self.cost.calculate(Y, Yhat))
            for metric in self.metrics:
                self.history[metric.name].append(metric.calculate(Y, Yhat))

            # Backward Propagation
            self.backward(Y, Yhat)
            
            pbar.set_description(self.history)
        
        return self.history
    
    def forward(self, X):
        Ai = X # X is A0, forward(X) is A1, .. Yhat is Al
        for layer in self.layers:
            Ai = layer.forward(Ai)
        Yhat = Ai
        return Yhat
    
    def backward(self, Y, Yhat):
        dAi = self.cost.derivative(Y, Yhat)
        for layer in self.layers[::-1]:
            dAi, dW = layer.backward(dAi)


            




        