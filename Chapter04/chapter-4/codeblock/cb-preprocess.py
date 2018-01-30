import numpy as np


def preprocess_input(X):
    mean_X = np.mean(X, axis=0)
    centered_X = X - mean_X
    std_X = np.std(centered_X, axis=0)
    normalized_X = centered_X / std_X
    # Compute the covariance matrix from the centered data
    cov_matrix = np.dot(centered_X.T, centered_X) / centered_X.shape[0]
    # Perform Singular Valued Decomposition
    U,S,V = np.linalg.svd(cov_matrix)
    # Compute the whitened data without dimensionality reduction
    decorr_X = np.dot(centered_X, U) # decorrelate the data
    # Compute the whitened data
    whitened_X = decorr_X / np.sqrt(S + 1e-5)
    return whitened_X

if __name__ == "__main__":
    X = np.random.random((32, 32))
    WX = preprocess_input(X)
    print WX
