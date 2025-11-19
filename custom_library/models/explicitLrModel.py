import numpy as np

from ..core import BaseLrModel
from ..layers import DenseLayer, ActivationLayer
from ..utils import metrics

class ExplicitLrModel(BaseLrModel):
    def fit(self, X_train, y_train, X_eval, y_eval, epochs=1000, batch_size=32, learning_rate=0.01):
        """train models"""
        n_samples = X_train.shape[0]

        for epoch in range(epochs):
            for batch_start in range(0, n_samples, batch_size):
                batch_end = min(batch_start + batch_size, n_samples)

                 # Fetch the batch
                y_true_batch = y_train[batch_start:batch_end]
                X_batch = X_train[batch_start:batch_end]

                # Forward pass
                a = np.array(X_batch)
                zal = [a] # to keep both z or a


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

                    # keep forward data
                    zal.append(a)

                # Get zi
                y_pred_batch = a

                # dL/dz (MSE) = (2/N)*(zi-yi)
                dL_dz = (2/batch_size) * (y_pred_batch - y_true_batch)

                # Set iterator
                i = len(zal)-1 

                # Compute gradients and Update weights
                for layer in reversed(list(self.layers)): 
                    # cause i=max() have done calculate at first dL/dz
                    i-=1

                    if(isinstance(layer, DenseLayer)):                
                        # z(l) = a(l-1) * W(l) + b(l)
                        # a(l-1) -> z(l) -> a(l) -> L
                        # compute gradients w.r.t weights and biases
                        # dL/dW = dz/dw * dL/dz
                        # dL/dW = a^T * dL_dz
                        #
                        # dL/db = dz/db * dL/dz
                        # dL/db = [1]^T * dL/dz
                        # dL/db = sum(dL/dz)
                        #
                        # dL/dw(l-1) = dz/dw(l-1) * dL/dz(l-1)
                        # dL/dw(l-1) = a(l-2)^T * dL/dz(l-1)

                        dL_dW = np.dot(zal[i].T, dL_dz)
                        dL_db = np.sum(dL_dz, axis=0, keepdims=True)
                        

                        # Update weights and biases
                        layer.W -= learning_rate * dL_dW
                        layer.b -= learning_rate * dL_db

                        # Not has activate function yet
                        dL_dz = np.dot(dL_dz, layer.W.T)

                    if(isinstance(layer, ActivationLayer)):
                        # compute dL_dz for the next layer (if any)
                        # z(l-1) -> a(l-1) -> z(l) -> L
                        # dL/da(l-1) = dz(l)/da(l-1) * dL/dz(l) 
                        # dL/da(l-1) =  dL/dz(l) * W^T
                        #
                        # dL/dz(l-1) = da(l-1)/dz(l-1) * dL/da(l-1)
                        # dL/dz(l-1) = f'(z(l-1)) dot dL/da(l-1)
                        dL_dz = layer.backward(zal[i]) * dL_dz

            # Predict
            pred_train = self.predict(X_train)
            pred_eval = self.predict(X_eval)

            # Evaluate
            train_acc = metrics.mae(y_train, pred_train)
            val_acc = metrics.mae(y_eval, pred_eval)
            
            # Save
            self.history.save(train_acc, val_acc)
            self.history.save_predict(X_eval, pred_eval, y_eval)
            self.history.save_model(self.layers, val_acc)

            print(f"Epoch {epoch+1}/{epochs} [", end="")

            progress_bar_length = 25
            progress = int((epoch/epochs)*progress_bar_length)
            for i in range(progress_bar_length):
                if(i<=progress):
                    print("=", end="")
                else:
                    print(".", end="")
            print("]")
            print(f"loss: {train_acc:.4f}, val_loss: {val_acc:.4f}")
            print("") 
        print(f"best-loss: {self.history.get_best_loss():.4f}")
        return self.history