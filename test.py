import math

import pandas as pd
import numpy as np

path = 'Concrete_Data.csv'
df = pd.read_csv(path, sep=',', header=0)
for column in df.columns:
    df[column] = (df[column] - df[column].min())
    df[column] = df[column] / df[column].max()
labels = df['Strength']
labels = labels.apply(lambda x: min(9, math.floor(10 * x)))
labels_frame = pd.Series.to_frame(labels, name = '# label')
df = df.drop(['Strength'], axis=1)
labels_frame.to_csv('concrete-labels.csv', index=False)
df.to_csv('concrete-src.csv', sep=';', index=False)