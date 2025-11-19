import numpy as np

class DenseLayer:
    def __init__(self, input_dim, output_dim):
        self.W = np.random.randn(input_dim, output_dim) * np.sqrt(2/input_dim)
        self.b = np.random.randn(1, output_dim) * 0.01 
        self.input_dim = input_dim
        self.output_dim = output_dim

    def params_count(self):
        return self.input_dim * self.output_dim + self.output_dim