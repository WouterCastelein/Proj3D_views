import os
import sys
from glob import glob

import MulticoreTSNE
import numpy as np
import pandas as pd
import umap
from sklearn import decomposition, manifold, random_projection

import ae
import constants
import tapkee
from metrics import (compute_distance_list, distance_list_to_matrix,
                     metric_continuity, metric_neighborhood_hit,
                     metric_normalized_stress,
                     metric_shepard_diagram_correlation,
                     metric_trustworthiness)


def compute_metrics(X_high, X_low, D_high_l, D_low_l):
    D_high_m = distance_list_to_matrix(D_high_l)
    D_low_m = distance_list_to_matrix(D_low_l)

    T = metric_trustworthiness(X_high, X_low, D_high_m.copy(), D_low_m.copy())
    C = metric_continuity(X_high, X_low, D_high_m.copy(), D_low_m.copy())
    S = metric_shepard_diagram_correlation(X_high, X_low, D_high_l, D_low_l)
    N = metric_normalized_stress(X_high, X_low, D_high_l, D_low_l)

    print('T: {0}, C: {1}, S: {2}, N: {3}'.format(T, C, S, N))

    return T, C, S, N

if __name__ == '__main__':
    dataset_name = sys.argv[1]
    input_file = glob('CAG/{0}/*-src.csv'.format(dataset_name))[0]

    metrics_file = 'CAG/metrics_{0}.csv'.format(dataset_name)
    output_dir = 'CAG/projections'

    df = pd.read_csv(input_file, sep=';', header=0)
    X_high = df.to_numpy()
    D_high_l = compute_distance_list(X_high)

    dataset = input_file.split('/')[1]
    projections = glob(os.path.join(constants.output_dir, '{0}*.csv'.format(dataset_name)))

    metrics_list = []

    for proj_file in projections:
        elems = proj_file.split('-')[1:]
        n_components = int(elems[-1].replace('d.csv', ''))
        proj_name = '-'.join(elems[::-1][1:][::-1])

        print('file: {0}, dim: {1}, proj: {2}'.format(proj_file, n_components, proj_name))

        df_low = pd.read_csv(proj_file, sep=';', header=0)
        X_low = df_low.to_numpy()
        D_low_l = compute_distance_list(X_low)

        T, C, S, N = compute_metrics(X_high, X_low, D_high_l, D_low_l)
        metrics_list.append((proj_name, n_components, T, C, S, N))


    df_metrics = pd.DataFrame.from_records(metrics_list)
    df_metrics.columns = ['projection_name', 'n_components', 'trustworthiness', 'continuity', 'shepard_correlation', 'normalized_stress']
    df_metrics.to_csv(metrics_file, index=None, sep=';')

    # df = pd.read_csv(input_file, sep=';', header=0)
    # X_high = df.to_numpy()

    # selected_projections = projection_names[selection]

    # header = ['x', 'y', 'z']

    # for n_components in [2, 3]:
    #     projections = get_projections(n_components)

    #     for proj_name in selected_projections:
    #         p = projections[proj_name]
    #         output_file = os.path.join(output_dir, '{0}-{1}-{2}d.csv'.format(dataset_name, proj_name, n_components))

    #         print('dim: {0}, proj: {1}'.format(n_components, proj_name))
    #         print('output_file: {0}'.format(output_file))

    #         X_low = p.fit_transform(X_high)

    #         df_low = pd.DataFrame(X_low)
    #         df_low.columns = header[:n_components]

    #         df_low.to_csv(output_file, index=None, sep=';')
