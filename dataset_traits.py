import os
from glob import glob
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from scipy import spatial
from scipy import stats

def metric_dc_num_samples(X):
    return X.shape[0]

def metric_dc_num_features(X):
    return X.shape[1]

def metric_dc_sparsity_ratio(X):
    return 1.0 - (np.count_nonzero(X) / float(X.shape[0] * X.shape[1]))

def metric_dc_intrinsic_dim(X):
    pca = PCA()
    pca.fit(X)

    return np.where(pca.explained_variance_ratio_.cumsum() >= 0.95)[0][0] + 1

if __name__ == '__main__':
    data_files = glob('data/**/*-src.csv', recursive=True)

    metrics = []

    for d in data_files:
        dataset_name = os.path.dirname(d).split('\\')[1]
        df = pd.read_csv(d, sep=';', header=0)
        X = df.to_numpy()

        N = metric_dc_num_samples(X)
        n = metric_dc_num_features(X)
        gamma = metric_dc_sparsity_ratio(X)
        rho = metric_dc_intrinsic_dim(X)

        dataset_type = 'tables'

        if dataset_name == 'Reuters':
            dataset_type = 'text'

        metrics.append((dataset_name, dataset_type, N, n, gamma, rho))


    df_metrics = pd.DataFrame.from_records(metrics)
    df_metrics.columns = ['dataset_name', 'dataset_type', 'size', 'dimensionality', 'sparsity_ratio', 'intrinsic_dimensionality']
    df_metrics.to_csv('dataset_metrics.csv', index=None, sep=';')
