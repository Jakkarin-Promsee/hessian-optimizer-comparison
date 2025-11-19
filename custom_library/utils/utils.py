import numpy as np

def split_data(X, y, split_ratio=0.8):
    # 1. Get the number of samples
    num_samples = X.shape[0]

    # 2. Create a shuffled index array
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    # 3. Calculate the split point
    split_point = int(num_samples * split_ratio)

    # 4. Split the indices
    train_indices = indices[:split_point]
    eval_indices = indices[split_point:]

    # 5. Use the indices to get the training and evaluation data
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_eval = X[eval_indices]
    y_eval = y[eval_indices]

    return X_train, y_train, X_eval, y_eval

def sort_data(y_pred, y_true, by="true"):
    y_all = np.column_stack((y_pred, y_true))

    if by == "true":
        sort_idx = np.argsort(y_all[:, 1])  # sort by y_true
    else:
        sort_idx = np.argsort(y_all[:, 0])  # sort by y_pred

    y_all_sorted = y_all[sort_idx]

    return y_all_sorted[:, 0], y_all_sorted[:, 1]
