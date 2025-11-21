import copy
import numpy as np
from typing import List, Union

from ..layers import DenseLayer, ActivationLayer
from ..utils import History, metrics

class BaseLrModel:
    def __init__(self):
        self.history = History()

        # Using type hints.
        self.layers: List[Union[DenseLayer, ActivationLayer]] = []
        self.last_output_dim = None

        print('test')

    def getHistory(self):
        return self.history
    
    def copylayers(self, layers):
        self.layers = copy.deepcopy(layers)
    
    def add(self, dense, activation="", input_shape=None):
        """Add layer to models. First adding should provide input_shape"""
        # Set input_shape
        if(self.last_output_dim == None and input_shape == None):
            raise ValueError("Input shape missing")
        elif(self.last_output_dim == None):
            self.last_output_dim = input_shape

        # Create Dense layer
        layer = DenseLayer(self.last_output_dim, dense)
        self.layers.append(layer)
        self.last_output_dim = dense

        # Create Activation layer
        if(activation!=""):
            layer = ActivationLayer(activation)
            self.layers.append(layer)

    def add_activation(self, method):
        """Add activation layer"""
        layer = ActivationLayer(method)
        self.layers.append(layer)

    def load(self, history=History()):
        """"""
        self.history = history
        self.layers = history.get_layers()

    def total_params(self):
        """Get totoal parameters (int)"""
        return sum(layer.params_count() for layer in self.layers if isinstance(layer, DenseLayer)) 
    
    def rollback(self):
        self.layers = self.history.get_best_layers()
    
    def predict_best(self, X):
        self.layers = self.history.get_best_layers()
        pred = self.predict(X)
        self.layers = self.history.get_layers()

        return pred
    
    def predict(self, X):
        """Get prediction from single point / array point"""
        # Intial first a to forward
        a = X

        for layer in self.layers:
            # If now we're on dense layer
            if(isinstance(layer, DenseLayer)):
                if(a.shape[1] != layer.input_dim):
                    raise ValueError("Input dimension mismatch: expected " + 
                                     f"{layer.input_dim}, got {a.shape[1]}")
            
                # z = X * W + b
                # z = a ; no activate funciton
                a = np.dot(a, layer.W) + layer.b
            
            # If now we're on activation layer
            if(isinstance(layer, ActivationLayer)):
                # a = f(z) which now z = a
                a = layer.forward(a)

        # Return last layer output
        return a

    def fit(self, X, y, **kwargs):
        # A simple base fit method can raise an error or just pass
        raise NotImplementedError("The 'fit' method must be implemented by the subclass.")