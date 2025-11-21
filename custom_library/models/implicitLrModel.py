import copy
import numpy as np

from ..core import BaseLrModelsImplementHessain
from ..layers import DenseLayer, ActivationLayer
from ..utils import metrics

class ImplicitLrModel(BaseLrModelsImplementHessain):
    def fit(self, X_train, y_train, X_eval, y_eval, epochs=100, batch_size=32, learning_rate=0.1, epsilon=1e-5):
        """train models"""
        n_samples = X_train.shape[0]

        for epoch in range(epochs):
            for batch_start in range(0, n_samples, batch_size):
                batch_end = min(batch_start + batch_size, n_samples)

                 # Fetch the batch
                y_true_batch = y_train[batch_start:batch_end]
                X_batch = X_train[batch_start:batch_end]

                # Backward euler on a quadratic method
                theta, shapes = self.flatten_params(self.layers)
                g = self.compute_gradient(self.layers, X_batch, y_true_batch)
                H = self.compute_hessian(self.layers, X_batch, y_true_batch, epsilon=epsilon)  
                
                # (I + eta H)
                H_i = np.eye(H.shape[0]) + learning_rate*H

                # (theta + eta b)
                theta_term = theta + learning_rate*(np.dot(H, theta) - g)

                # Calculate quadratic step direction
                try:
                    # theta_k+1 = (I + eta H)^-1 (theta_k + eta b)
                    # Solve: (I + eta H) * delta = (theta + eta b)
                    delta = np.linalg.solve(H_i, theta_term)
                except np.linalg.LinAlgError:
                    print("Warning: Hessian is singular. Skipping step.")
                    continue  

                # --- Update Weight (using the accepted theta_new) ---
                self.unflatten_params(self.layers, delta, shapes)

                # Predict
                pred_batch = self.predict(X_batch)
                pred_train = self.predict(X_train)
                pred_eval = self.predict(X_eval)

                # Evaluate
                batch_acc = metrics.mae(y_true_batch, pred_batch)
                train_acc = metrics.mae(y_train, pred_train)
                val_acc = metrics.mae(y_eval, pred_eval)
                
                # Save
                self.history.save(train_acc, val_acc)
                self.history.save_predict(X_eval, pred_eval, y_eval)
                self.history.save_model(self.layers, val_acc)

                # Log batch data
                print(f"Epoch {epoch+1}/{epochs} [", end="")

                progress_bar_length = 25
                progress = int((epoch/epochs)*progress_bar_length)
                for i in range(progress_bar_length):
                    if(i<=progress):
                        print("=", end="")
                    else:
                        print(".", end="")
                print("]", end=", ")

                print(f"{batch_start}/{n_samples}: ")
                print(f"batch: {batch_acc:.3f}, acc: {train_acc:.3f}, val: {val_acc:.3f}")

        print(f"best-loss: {self.history.get_best_loss():.4f}")
        return self.history