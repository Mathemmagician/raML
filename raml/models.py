
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
    
    def iterate_minibatches(self, X, Y, batchsize, shuffle=False):
        '''Yields mini batches
        
        Example Usage
        -------------
        for n in xrange(n_epochs):
            for batch in iterate_minibatches(X, Y, batch_size, shuffle=True):
                x_batch, y_batch = batch
                ...
        '''
        assert X.shape[1] == Y.shape[1]
        if shuffle:
            indices = np.arange(X.shape[1])
            np.random.shuffle(indices)
        for start_idx in range(0, X.shape[1] - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            yield X[:,excerpt], Y[:,excerpt]


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
    

    def fit(self, X, Y, epochs, x_val=None, y_val=None, batchsize=32):
        '''
        Params:
            X : np.ndarray  | input data
            Y : np.ndarray  | output data
            epochs : int    | number of training steps
        '''
        
        if batchsize is None:
            batchsize = X.shape[1]
        
        self.history = {each : [] for each in ["Loss"] + [metric.name for metric in self.metrics]}
        
        if (validation := (x_val is not None) and (y_val is not None)):
            for each in ["Loss"] + [metric.name for metric in self.metrics]:
                self.history[f'val_{each}'] = []

        for i in (pbar := raml_tqdm(range(epochs))):

            for batch in self.iterate_minibatches(X, Y, batchsize=batchsize, shuffle=False):
                X_batch, Y_batch = batch

                if validation:
                    Yhat_val = self.forward(x_val)
                    self.history["val_Loss"].append(self.cost.calculate(y_val, Yhat_val))
                    for metric in self.metrics:
                        self.history[f"val_{metric.name}"].append(metric.calculate(y_val, Yhat_val))

                # Forward Propagation
                Yhat_batch = self.forward(X_batch)

                # Computing and recording the loss
                self.history["Loss"].append(self.cost.calculate(Y_batch, Yhat_batch))
                for metric in self.metrics:
                    self.history[metric.name].append(metric.calculate(Y_batch, Yhat_batch))

                # Backward Propagation
                self.backward(Y_batch, Yhat_batch)
            
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


            




        