import numpy as np

def mae(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    return mae

def mse(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    return mse

def rmse(y_true, y_pred):
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return rmse

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2

def accuracy(y_true, y_pred):
    maes = mae(y_true, y_pred)
    accuracy = 1 - maes / (y_true.max() - y_true.min())
    return maes 