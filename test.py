import numpy as np
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import pickle
import compute_metrics
import metrics

input_file = f"evaluationdata/evaluationdata Kirsten Maas.pkl"
input_file_2 = f"evaluationdata/evaluationdata user (this time anonymous).pkl"
df1 = pd.read_pickle(input_file)
df2 = pd.read_pickle(input_file_2)
print()