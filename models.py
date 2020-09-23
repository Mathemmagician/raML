
from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np
from tqdm import tqdm

class Model:
    def __init__(self):
        pass

    def compile(self, cost, optimizer = None):
        self.cost = cost
        self.optimizer = optimizer

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
        super().compile(*args, **kwargs)
        

    def fit(self, X, Y, epochs, epochstep):
        '''
        Params:
            X : np.ndarray  | input data
            Y : np.ndarray  | output data
            epochs : int    | number of training steps
            epochstep : int | how often the progress will be displayed
        '''

        self.history = {"loss": []}

        for i in (pbar := tqdm(range(epochs))):

            # Forward Propagation
            Yhat = self.forward(X)

            # Computing and recording the loss
            loss = self.cost.calculate(Y, Yhat)
            self.history["loss"].append(loss)

            # Backward Propagation
            self.backward(Y, Yhat)
            
            # if i % epochstep == 0:
            #     print(f'Iteration {i} : {self.cost}')
            pbar.set_description(f"Loss {loss:.5f}")
    
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


            




        