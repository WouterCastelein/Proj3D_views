import numpy as np
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import pickle

with open('evaluationdata.pkl', 'rb') as file:
    a = pickle.load(file)