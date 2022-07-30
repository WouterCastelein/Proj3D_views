import numpy as np
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import pickle
import compute_metrics
import metrics

i = np.load('initial.npy')
o = np.load('opposite.npy')

input_file = f'data/{"Concrete"}/{"concrete"}-src.csv'
df = pd.read_csv(input_file, sep=';', header=0)
X_high = df.to_numpy()
D_high_l = metrics.compute_distance_list(X_high)

X_low = i
D_low_l1 = metrics.compute_distance_list(X_low)
print(compute_metrics.compute_metrics(X_high, X_low, D_high_l, D_low_l1))
X_low = o
D_low_l2 = metrics.compute_distance_list(X_low)
print(compute_metrics.compute_metrics(X_high, X_low, D_high_l, D_low_l2))