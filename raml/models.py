
from abc import ABC, abstractmethod
from collections import defaultdict
import pickle
import os

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
        while True: # Hack for infinite batch generation
            for start_idx in range(0, X.shape[1] - batchsize + 1, batchsize):
                if shuffle:
                    excerpt = indices[start_idx:start_idx + batchsize]
                else:
                    excerpt = slice(start_idx, start_idx + batchsize)
                yield X[:,excerpt], Y[:,excerpt]
    
    def forward(self, X):
        pass

    def backward(self, X):
        pass
    
    def predict(self, X):
        '''CLASSIFICATION ONLY. For each sample returns the index of label with max probability'''
        return np.argmax(self.forward(X), axis = 0)

    def evaluate(self, X, Y):
        '''CLASSIFICATION ONLY. Returns model's accuracy, i.e. how many predicted '''
        return (self.predict(X) == np.argmax(Y, axis = 0)).mean()
    
    def save(self, filename = None):
        '''Needs work'''
        if filename is None:
            filename = f'model.obj'

        with open(os.path.join('raml_cache', filename), 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename = None):
        '''Needs work'''
        if filename is None:
            filename = f'model.obj'
        
        with open(os.path.join('raml_cache', filename), 'rb') as f:
            model = pickle.load(f)
        
        return model


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
        Params
        ------
            X : np.ndarray
                training data
            Y : np.ndarray
                training labels
            epochs : int
                number of training steps
            x_val : np.ndarray
                validation data
            y_val : np.ndarray
                validation labels
            batchsize : int
                size of mini-batches

        Returns
        -------
            self.history : dict
                This needs work

        Notes
        -----
            
        '''
        
        if batchsize is None:
            batchsize = X.shape[1]
        
        self.history = defaultdict(list)
        validate = (x_val is not None) and (y_val is not None)
        validation_step = X.shape[1] // batchsize

        batch_generator = self.iterate_minibatches(X, Y, batchsize=batchsize, shuffle=False)

        for epoch in (pbar := raml_tqdm(range(epochs))):
            X_batch, Y_batch = batch = next(batch_generator)

            # Recording validation loss
            if validate:
                if (epoch % validation_step == 0) or (epoch == epochs-1):
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
