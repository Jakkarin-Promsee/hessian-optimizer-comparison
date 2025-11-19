import numpy as np

class ActivationLayer:
    def __init__(self, method):
        if(method=="relu"):
            self.activate_func = self.relu
            self.activate_derivative = self.relu_derivative
        else:
            self.activate_func = None
            self.activate_derivative = None

    def forward(self, z):
        if(self.activate_func is not None):
            return self.activate_func(z)
        return z
    
    def backward(self, z):
        if(self.activate_derivative is not None):
            return self.activate_derivative(z)
        return z
    
    # ----------------------------- Method Part ----------------------------

    def relu(self, x):
        f = np.copy(x)
        f[x<0] = 0
        return f
    
    def relu_derivative(self, x):
        g = np.zeros_like(x)
        g[x > 0] = 1.0      # subgradient 0 at x==0
        return g