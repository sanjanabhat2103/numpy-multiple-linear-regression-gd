"""
NumPy Multiple Linear Regression GD

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - shuffle_xy
def shuffle_xy(X, y, seed=42):
    """Randomly permute feature rows and targets together.

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        Feature matrix.
    y : np.ndarray, shape (n,)
        Target vector.
    seed : int, optional
        RNG seed for reproducibility (default 42).

    Returns
    -------
    X_shuffled : np.ndarray, shape (n, d)
    y_shuffled : np.ndarray, shape (n,)
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X))
    return X[perm], y[perm]

# Step 2 - split_train_val_test
def split_train_val_test(X, y, train_frac=0.6, val_frac=0.2):
    n = len(X)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    X_train = X[: n_train]
    y_train = y[: n_train]
    X_val = X[n_train: n_train + n_val]
    y_val = y[n_train: n_train + n_val]
    X_test = X[n_train + n_val: ]
    y_test = y[n_train + n_val: ]
    return X_train, y_train, X_val, y_val, X_test, y_test

# Step 3 - compute_feature_stats
def compute_feature_stats(X):
    # TODO: Compute per-feature mean and std; replace std of 0 with 1
    mean = np.mean(X, axis = 0)
    std = np.std(X, axis = 0)
    std[std == 0] = 1
    return mean, std

# Step 4 - standardize_features
def standardize_features(X, mean, std):
    return (X - mean) / std

# Step 5 - add_bias_column
def add_bias_column(X):
    bias = np.ones((X.shape[0], 1))
    return np.hstack((bias, X))

# Step 6 - prepare_design_matrix
def prepare_design_matrix(X, mean, std):
    X_std = standardize_features(X, mean, std)
    return add_bias_column(X_std)

# Step 7 - predict_linear
def predict_linear(X, weights):
    """Compute linear predictions y_hat = X @ weights.

    Args:
        X: Design matrix of shape (n, d_in), often including a bias column.
        weights: Weight vector of shape (d_in,).

    Returns:
        Predicted targets of shape (n,).
    """
    return X @ weights

# Step 8 - mse_loss
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Step 9 - mse_gradient
def mse_gradient(X, y_true, y_pred):
    n = X.shape[0]
    r = y_pred - y_true
    return 2.0 / n * (X.T @ r)

# Step 10 - normal_equation
def normal_equation(X, y):
    return np.linalg.solve(X.T @ X, X.T @ y)

# Step 11 - initialize_weights
def initialize_weights(n_features, seed=None):
    rng = np.random.RandomState(seed)
    return rng.normal(loc = 0.0, scale = 0.01, size = n_features)

# Step 12 - gd_step
def gd_step(X, y, weights, lr):
    """Run one full-batch gradient descent update on the weights.

    Args:
        X: Design matrix of shape (n, d_in).
        y: Target vector of shape (n,).
        weights: Current weight vector of shape (d_in,).
        lr: Learning rate (float).

    Returns:
        Updated weight vector of shape (d_in,).
    """
    y_pred = predict_linear(X, weights)
    gradient = mse_gradient(X, y, y_pred)
    return weights - lr * gradient

# Step 13 - epoch_train_val_losses
def epoch_train_val_losses(X_train, y_train, X_val, y_val, weights):
    """Evaluate MSE on train and validation sets for the current weights.

    Args:
        X_train: Training design matrix of shape (n_tr, d_in).
        y_train: Training targets of shape (n_tr,).
        X_val: Validation design matrix of shape (n_va, d_in).
        y_val: Validation targets of shape (n_va,).
        weights: Weight vector of shape (d_in,).

    Returns:
        (train_loss, val_loss) as plain floats.
    """
    train_pred = predict_linear(X_train, weights)
    val_pred = predict_linear(X_val, weights)
    train_loss = mse_loss(y_train, train_pred)
    val_loss = mse_loss(y_val, val_pred)
    return float(train_loss), float(val_loss)

# Step 14 - update_early_stop_state
def update_early_stop_state(val_loss, best_val_loss, wait, weights, best_weights, patience):
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_weights = weights.copy()
        wait = 0
    else:
        wait += 1
    stop = wait >= patience
    return best_val_loss, wait, best_weights, stop

# Step 15 - init_training_state
def init_training_state(n_features, seed=None):
    weights = initialize_weights(n_features = n_features, seed = seed)
    return {
        "weights": weights,
        "best_weights": weights.copy(),
        "best_val_loss": np.inf,
        "wait": 0,
        "stopped": False,
        "train_losses": [],
        "val_losses": [],
    }

# Step 16 - run_one_epoch
def run_one_epoch(state, X_train, y_train, X_val, y_val, lr, patience):
    """Perform one GD step, log losses, and refresh early-stopping on state.

    Args:
        state: Dict with keys weights, best_weights, best_val_loss, wait,
            stopped, train_losses, val_losses.
        X_train: Training design matrix of shape (n_tr, d_in).
        y_train: Training targets of shape (n_tr,).
        X_val: Validation design matrix of shape (n_va, d_in).
        y_val: Validation targets of shape (n_va,).
        lr: Learning rate (float).
        patience: Early-stopping patience (int).

    Returns:
        Updated state dict.
    """
    state["weights"] = gd_step(X_train, y_train, state["weights"], lr)
    train_loss, val_loss = epoch_train_val_losses(X_train, y_train, X_val, y_val, state["weights"])
    state["train_losses"].append(train_loss)
    state["val_losses"].append(val_loss)
    (state["best_val_loss"], state["wait"], state["best_weights"], state["stopped"]) = update_early_stop_state(val_loss, state["best_val_loss"], state["wait"], state["weights"], state["best_weights"], patience)
    return state

# Step 17 - train_batch_gd
def train_batch_gd(X_train, y_train, X_val, y_val, lr, epochs, patience, seed=None):
    state = init_training_state(X_train.shape[1], seed = seed)
    for _ in range(epochs):
        state = run_one_epoch(state, X_train, y_train, X_val, y_val, lr, patience)
        if state["stopped"]:
            break
    return state['best_weights'], state['train_losses'], state['val_losses']

# Step 18 - mean_absolute_error
def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

# Step 19 - root_mean_squared_error
def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mse_loss(y_true, y_pred))

# Step 20 - r_squared (not yet solved)
# TODO: implement

# Step 21 - evaluate_regression (not yet solved)
# TODO: implement

# Step 22 - learning_curve_data (not yet solved)
# TODO: implement

# Step 23 - weights_l2_distance (not yet solved)
# TODO: implement

# Step 24 - create_lr_model (not yet solved)
# TODO: implement

# Step 25 - fit_lr_model (not yet solved)
# TODO: implement

# Step 26 - predict_lr_model (not yet solved)
# TODO: implement

# Step 27 - score_lr_model (not yet solved)
# TODO: implement

# Step 28 - compare_with_normal_equation (not yet solved)
# TODO: implement

