import numpy as np
import copy

from . import BaseLrModel
from ..layers import DenseLayer, ActivationLayer

class BaseLrModelsImplementHessain(BaseLrModel):
    def flatten_params(self, layers):
        flat_list = []
        shapes = []

        for layer in layers:
            if(isinstance(layer, ActivationLayer)):
                continue

            if hasattr(layer, "W"):
                shapes.append(("W", layer.W.shape))
                flat_list.append(layer.W.ravel())

            if hasattr(layer, "b"):
                shapes.append(("b", layer.b.shape))
                flat_list.append(layer.b.ravel())

        return np.concatenate(flat_list), shapes
    
    def unflatten_params(self, layers, theta_vec, shapes):
        i=0
        idx = 0
        for layer in layers:
            if(isinstance(layer, ActivationLayer)):
                continue
            
            if hasattr(layer, "W"):
                name, shape = shapes[i]
                size = np.prod(shape)
                layer.W = theta_vec[idx:idx+size].reshape(shape)
                idx += size
                i += 1

            if hasattr(layer, "b"):
                name, shape = shapes[i]
                size = np.prod(shape)
                layer.b = theta_vec[idx:idx+size].reshape(shape)
                idx += size
                i += 1

    # New method to add to NewtonLrModel class:
    def compute_loss(self, layers, X, y_true):
        # Forward pass logic (can reuse compute_gradient's forward pass)
        a = np.array(X)
        for layer in layers:
            if(isinstance(layer, DenseLayer)):
                a = np.dot(a, layer.W) + layer.b
            if(isinstance(layer, ActivationLayer)):
                a = layer.forward(a)
        y_pred = a
        
        # Calculate MSE Loss: (1/N) * sum((y_pred - y_true)^2)
        N = X.shape[0]
        loss = np.sum((y_pred - y_true)**2) / N
        return loss

    def compute_gradient(self, layers, X, y_true):
        # Forward pass
        a = np.array(X)
        zal = [a] # to keep both z or a

        for layer in layers:
            # If now we're on dense layer
            if(isinstance(layer, DenseLayer)):
                if(a.shape[1] != layer.input_dim):
                    raise ValueError("Input dimension mismatch: expected " + 
                                    f"{layer.input_dim}, got {a.shape[1]}")
                a = np.dot(a, layer.W) + layer.b

            # If now we're on activation layer
            if(isinstance(layer, ActivationLayer)):
                a = layer.forward(a)

            # keep forward data
            zal.append(a)

        # Get zi
        y_pred = a

        # dL/dz (MSE) = (2/N)*(zi-yi)
        dL_dz = (2/X.shape[0]) * (y_pred - y_true)

        # Set iterator
        i = len(zal)-1 

        gradient = []

        # Compute gradients and Update weights
        for layer in reversed(list(layers)): 
            # cause i=max() have done calculate at first dL/dz
            i-=1

            if(isinstance(layer, DenseLayer)):           
                # Compute Gradient     
                dL_dW = np.dot(zal[i].T, dL_dz)
                dL_db = np.sum(dL_dz, axis=0, keepdims=True)

                # Store Gradient
                gradient.insert(0, dL_db)
                gradient.insert(0,dL_dW)
                
                
                # Compute next delta
                dL_dz = np.dot(dL_dz, layer.W.T)

            if(isinstance(layer, ActivationLayer)):
                dL_dz = layer.backward(zal[i]) * dL_dz

        return np.concatenate([g.ravel() for g in gradient])

    
    def compute_hessian(self, layers, X, y, epsilon=1e-2):
        theta, shapes = self.flatten_params(layers)
        
        N = theta.size
        H = np.zeros((N, N))

        for j in range(N):
            e = np.zeros(N)
            e[j] = 1

            # theta + eps
            theta_pos = theta + epsilon * e
            layers_pos = copy.deepcopy(layers)
            self.unflatten_params(layers_pos, theta_pos, shapes)
            g_pos = self.compute_gradient(layers_pos, X, y)

            # theta - eps
            theta_neg = theta - epsilon * e
            layers_neg = copy.deepcopy(layers)
            self.unflatten_params(layers_neg, theta_neg, shapes)
            g_neg = self.compute_gradient(layers_neg, X, y)

            # second derivative column j
            H[:, j] = (g_pos - g_neg) / (2 * epsilon)
        return H