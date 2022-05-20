import os
import pandas as pd
from glob import glob

metrics_dir = 'CAG'
consolid_metrics_file = os.path.join(metrics_dir, 'metrics.csv')
metrics_files = glob(os.path.join(metrics_dir, 'metrics_*.csv'))

dfs = []
datasets = []
columns = []

for m in metrics_files:
    df = pd.read_csv(m, sep=';', header=0)
    dfs.append(df)
    datasets += [os.path.basename(m).split('_')[1].replace('.csv', '') for i in range(len(df))]
    columns = df.columns

df_metrics = pd.concat(dfs)
df_metrics.columns = columns
df_metrics['dataset_name'] = datasets
df_metrics = df_metrics.loc[:,['dataset_name', 'projection_name', 'n_components', 'trustworthiness', 'continuity', 'shepard_correlation', 'normalized_stress']]

df_metrics.to_csv(consolid_metrics_file, index=None, sep=';')
