import multiprocessing
import os
import sys
from glob import glob
from functools import partial
import numpy as np
import pandas as pd
import constants
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

    #print('T: {0}, C: {1}, S: {2}, N: {3}'.format(T, C, S, N))

    return T, C, S, N

def parallel_metrics(X_high, D_high_l, args):
    index, view = args
    D_v_low = compute_distance_list(view)
    T_v, C_v, S_v, N_v, = compute_metrics(X_high, view, D_high_l, D_v_low)
    if N_v < 0:
        print('below 0')
    if index % 10 == 0:
        print(f"Calculating approximately view: {index}")
    return [index, T_v, C_v, S_v, N_v]

if __name__ == '__main__':
    for dataset_name in ['Wine', 'Software', 'Reuters', 'Concrete', 'AirQuality']:
        print(dataset_name)
        input_file = glob(f'data/{dataset_name}/*-src.csv')[0]

        metrics_file = os.path.join(constants.metrics_dir, F'metrics_{dataset_name}.pkl')

        df = pd.read_csv(input_file, sep=';', header=0)
        X_high = df.to_numpy()
        D_high_l = compute_distance_list(X_high)

        dataset = input_file.split('/')[1]
        projections = glob(os.path.join(constants.output_dir, '{0}*.csv'.format(dataset_name)))

        metrics_list = []
        pool = multiprocessing.Pool(max(1, 4))
        for proj_file in projections:
            elems = proj_file.split('-')[1:]
            n_components = int(elems[-1].replace('d.csv', ''))
            proj_name = '-'.join(elems[::-1][1:][::-1])

            print('file: {0}, dim: {1}, proj: {2}'.format(proj_file, n_components, proj_name))

            df_low = pd.read_csv(proj_file, sep=';', header=0)
            X_low = df_low.to_numpy()
            D_low_l = compute_distance_list(X_low)

            T, C, S, N = compute_metrics(X_high, X_low, D_high_l, D_low_l)

            #Repeat for all views of a 3D projection:
            metrics_views_list = []
            if n_components == 3:
                views_file = proj_file.replace('3d.csv', 'views.pkl')
                views = pd.read_pickle(views_file)['views'].to_numpy()

                #Use multiprocessing to speed up metric calculation for the views

                metrics_views_list = pool.map(partial(parallel_metrics, X_high, D_high_l), zip(list(range(len(views))), views))
            metrics_list.append((proj_name, n_components, T, C, S, N, np.array(metrics_views_list)))


        df_metrics = pd.DataFrame.from_records(metrics_list)
        df_metrics.columns = ['projection_name', 'n_components', 'trustworthiness', 'continuity', 'shepard_correlation', 'normalized_stress', 'views_metrics']
        df_metrics.to_pickle(metrics_file)
