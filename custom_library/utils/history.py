import numpy as np
import matplotlib.pyplot as plt

class History:
    def __init__(self):
        # Each entry = [train_acc, val_acc]
        self.history = []

        # Each entry = [pred, y_true]
        self.last_predict = []
        self.last_X = []
       
        # Save best weight
        self.layers = []
        self.best_layers = []
        self.val_acc = None

    def save_model(self, layers, val_acc):
        """Save model weights"""
        self.layers = layers

        if(self.val_acc == None):
            self.best_layers = layers
            self.val_acc = val_acc
            return

        if(self.val_acc > val_acc):
            self.best_layers = layers
            self.val_acc = val_acc
    
    def get_layers(self):
        return self.layers
    
    def get_best_layers(self):
        return self.best_layers

    def get_best_loss(self):
        return self.val_acc

    def save(self, train_acc, val_acc):
        """Store training and validation accuracy per epoch"""
        self.history.append(np.array([train_acc, val_acc]))

    def save_predict(self, X, predict, y_true):
        """Store predictions and corresponding true values"""
        predict = np.array(predict).flatten()
        y_true = np.array(y_true).flatten()
        
        self.last_X = X
        self.last_predict = [predict, y_true]

    def display_trend(self, ref="x", axis=0, sort=True, show_X=False):
        """Visualize the trend of predictions vs true values. ref="x"||"y", x for input data set, y for expect data set"""
        if not self.last_predict:
            print("No predictions to display.")
            return

        X = np.asarray(self.last_X)
        y_pred, y_true = self.last_predict

        if sort:
            if(ref=="y"):
                sort_idx = np.argsort(y_true)
            else:
                sort_idx = np.argsort(X if X.ndim == 1 else X[:, axis])

            X = X[sort_idx]
            y_true = y_true[sort_idx]
            y_pred = y_pred[sort_idx]

        i = np.arange(len(y_true))
        if(show_X):
            plt.scatter(i, X if X.ndim == 1 else X[:, axis], alpha=0.5, color='yellow', label='T')

        plt.scatter(i, y_pred, alpha=0.4, color='red', label='Predictions')
        plt.scatter(i, y_true, alpha=0.4, color='blue', label='True Values')
        plt.title("Prediction Trend")
        plt.xlabel("Sample Index (sorted by true value)")
        plt.ylabel("Value")
        plt.legend()
        plt.show()

    def evaluate(self, start=0, end=None):
        """Plot training and validation accuracy history"""
        if not self.history:
            print("No training history available.")
            return

        hist_arr = np.array(self.history)
        if end is None:
            end = len(hist_arr)

        train_acc = hist_arr[start:end, 0]
        val_acc = hist_arr[start:end, 1]

        plt.plot(train_acc, label='Train Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training History')
        plt.legend()
        plt.show()

        y_pred, y_true = self.last_predict
        plt.scatter(y_true, y_pred)
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.plot([y_true.min(), y_true.max()], [y_pred.min(), y_pred.max()], 'r--')
        plt.title('True vs Predicted Values')
        plt.show()