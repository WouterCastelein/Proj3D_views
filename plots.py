#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
from glob import glob
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

import constants


def plot_3d(proj_file, output_dir='plots', labels=None):
    cmap = plt.get_cmap('tab10')

    df = pd.read_csv(proj_file, sep=';', header=0)
    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')
    ax2d = fig.add_subplot(122)
    if labels is not None:
        for cl in np.unique(labels):
            ax.scatter(df.loc[labels==cl,'x'], df.loc[labels==cl,'y'], df.loc[labels==cl,'z'], s=5, c=[cmap(cl)], label=cl)
            ax2d.scatter(df.loc[labels==cl,'x'], df.loc[labels==cl,'y'], s=5, c=[cmap(cl)], label=cl)
    else:
        ax.scatter(df['x'], df['y'], df['z'], s=10)
        ax2d.scatter(df['x'], df['y'], s=10)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    ax.legend()
    figname = '{0}/{1}.png'.format(output_dir, os.path.basename(proj_file).replace('.csv', ''))
    # rotate the axes and update
    fig.savefig(figname)

def plot_2d(proj_file, output_dir='plots', labels=None):
    cmap = plt.get_cmap('tab10')

    df = pd.read_csv(proj_file, sep=';', header=0)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    if labels is not None:    
        for cl in np.unique(labels):
            print(proj_file, cl, df.loc[labels==cl,'x'])
            ax.scatter(df.loc[labels==cl,'x'], df.loc[labels==cl,'y'], s=5, c=[cmap(cl)], label=cl)
    else:
        ax.scatter(df['x'], df['y'], s=10)

    ax.set_xlabel('x')
    ax.set_ylabel('y')

    ax.set_xticklabels([])
    ax.set_yticklabels([])

    figname = '{0}/{1}.png'.format(output_dir, os.path.basename(proj_file).replace('.csv', ''))
    fig.savefig(figname)


if __name__ == '__main__':
    projections_2d = glob(os.path.join(constants.output_dir, '*2d.csv'))
    projections_3d = glob(os.path.join(constants.output_dir, '*3d.csv'))

    for proj_file in projections_2d:
        dataset_name = os.path.basename(proj_file).split('-')[0]
        label_file = glob('data/{0}/*-labels.csv'.format(dataset_name))

        if len(label_file) == 1:
            df_label = pd.read_csv(label_file[0], sep=';', header=0)
            labels = df_label.to_numpy()
        else:
            labels = None

        plot_2d(proj_file, labels=labels)

    for proj_file in projections_3d:
        dataset_name = os.path.basename(proj_file).split('-')[0]
        label_file = glob('data/{0}/*-labels.csv'.format(dataset_name))

        if len(label_file) == 1:
            df_label = pd.read_csv(label_file[0], sep=';', header=0)
            labels = df_label.to_numpy()
        else:
            labels = None

        plot_3d(proj_file, labels=labels)
