import os.path

import pandas as pd
import numpy as np
import sys
from glob import glob
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.colors import LinearSegmentedColormap
from functools import partial
import constants
import os
from matplotlib.lines import Line2D
from matplotlib import cm


class Scatter3D:
    def __init__(self, ax, df3d, labels):
        self.ax = ax
        self.df = df3d
        self.cmap = plt.get_cmap('tab10')
        self.labels = labels
        self.ax.set_title('3D projection')

    def draw(self):
        if self.labels is not None:
            self.ax.scatter(self.df['x'], self.df['y'], self.df['z'], s=10, c=[self.cmap(cl) for cl in self.labels])
        else:
            self.ax.scatter(self.df['x'], self.df['y'], self.df['z'], s=10)

        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')

        #Ensure equal aspect ratio:
        #max = self.df.max_points_in_bin().max_points_in_bin()
       # self.ax.set_xlim(-max, max)
        #self.ax.set_ylim(-max, max)
       # self.ax.set_zlim(-max, max)
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        self.ax.set_zticklabels([])

        self.ax.set_box_aspect((4, 4, 4), zoom=1)

    def get_rotation(self):
        return self.ax.elev, self.ax.azim

    def set_rotation(self, elev, azim):
        self.ax.view_init(elev=elev, azim=azim)

class Scatter2D:
    def __init__(self, ax, df3d, labels):
        self.ax = ax
        self.df = df3d
        self.cmap = plt.get_cmap('tab10')
        self.labels = labels
        self.ax.set_title('2D projection')
        self.ax.axis('equal')
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])

    def draw(self):
        if self.labels is not None:
            self.ax.scatter(self.df['x'], self.df['y'], s=10, c=[self.cmap(cl) for cl in self.labels])
        else:
            self.ax.scatter(self.df['x'], self.df['y'], s=10)

        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')


class Histogram:
    def __init__(self, ax, views_metrics, metrics_2d, metrics_3d, dataset_name, projection_method):
        self.ax = ax
        self.views_metrics = views_metrics
        self.metrics_3d = metrics_3d
        self.metrics_2d = metrics_2d
        self.active_metrics = constants.metrics[:]
        self.cmap = plt.get_cmap('tab10')

    def draw(self):
        self.update()
        #plt.show()

    def update(self, active_metrics=constants.metrics):
        self.active_metrics = active_metrics
        self.ax.clear()
        if len(self.active_metrics) == 0:
            return
        displayed_metric_indices = np.array([constants.metrics.index(metric) for metric in self.active_metrics]) + 1
        displayed_metric_names = [constants.metrics[index] for index in displayed_metric_indices - 1]

        #Find the minimum and maximum metric value that we plot
        hist_range = (min(np.min(self.views_metrics[:, displayed_metric_indices]), np.min(self.metrics_2d[displayed_metric_names]),
                          np.min(self.metrics_3d[displayed_metric_names])),
                      max(np.max(self.views_metrics[:, displayed_metric_indices]), np.max(self.metrics_2d[displayed_metric_names]),
                          np.max(self.metrics_3d[displayed_metric_names])))
        legend_elements = []
        for metric in self.active_metrics:
            j = constants.metrics.index(metric) + 1
            bins = np.histogram(self.views_metrics[:, j], 100, range=hist_range)
            bar = self.ax.bar(np.array([(bins[1][x] + bins[1][x + 1]) / 2 for x in range(len(bins[1]) - 1)])[bins[0] != 0], bins[0][bins[0] != 0], alpha=0.7,
                   width=(hist_range[1] - hist_range[0]) / 100,
                   color=self.cmap(j - 1), edgecolor='black', label=constants.metrics[j - 1])
            legend_elements.append(bar)
            self.ax.axvline(x=self.metrics_2d[constants.metrics[j - 1]], color=self.cmap(j - 1), linestyle=(0, (5, 10)))
            self.ax.axvline(x=self.metrics_3d[constants.metrics[j - 1]], color=self.cmap(j - 1), linestyle=(0, (3, 5, 1, 5)))

        legend_elements += [Line2D([0], [0], label='2D', c="black", linestyle=(0, (5, 10))),
                           Line2D([0], [0], label='3D', c="black",linestyle=(0, (3, 5, 1, 5)))]

        # Create the figure
        self.ax.legend(handles=legend_elements, loc="center left", bbox_to_anchor=(1, 0.8))

class MetricSphere:
    def __init__(self, ax, views_metrics, viewpoints, active_metrics=["trustworthiness"]):
        self.views_metrics = views_metrics
        self.viewpoints = np.load(f'spheres/sphere{constants.samples}_points.npy')
        self.ax = ax
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        self.ax.set_zticklabels([])
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')
        c = ["darkred", "red", "yellow", "green", "darkgreen"]
        v = [0, 0.2, 0.5, 0.8, 1]
        l = list(zip(v, c))
        self.cmap = LinearSegmentedColormap.from_list('rg', l, N=256)
        self.active_metrics = active_metrics
        self.cb = None


    def update(self, active_metrics):
        self.active_metrics = active_metrics
        self.draw()


    def draw(self):
        metric_index = constants.metrics.index(self.active_metrics[0]) + 1 #Todo active metrics is a list for now, because i might want to aggregate multiple metrics later
        hm = self.views_metrics[:, metric_index]
        self.ax.set_box_aspect((4, 4, 4))
        scatter = self.ax.scatter(self.viewpoints[:, 0], self.viewpoints[:, 1], self.viewpoints[:, 2], c=hm, cmap=self.cmap,
                             vmin=np.amin(hm), vmax=np.amax(hm), s=75, edgecolors=None)
        if self.cb != None:
            self.cb.remove() #reset colorbar
        self.cb = plt.colorbar(scatter, ax=self.ax)
        self.ax.set_title(constants.metrics[metric_index - 1])

    def get_rotation(self):
        return self.ax.elev, self.ax.azim

    def set_rotation(self, elev, azim):
        self.ax.view_init(elev=elev, azim=azim)

class Tool:
    def __init__(self, df, dataset_name):
        self.dataset_name = dataset_name
        self.df = df
        self.fig = plt.figure(1)
        self.ax_3d = self.fig.add_subplot(221, projection='3d')
        self.ax_2d = self.fig.add_subplot(222)
        self.ax_sphere = self.fig.add_subplot(223, projection='3d')
        self.ax_histogram = self.fig.add_subplot(2, 2, 4)
        self.button_axes = [plt.axes([0.91, 0.05 + 0.065 * i, 0.08, 0.06]) for i in range(len(constants.metrics))]
        self.buttons = []
        self.cmap = plt.get_cmap('tab10')
        self.active_metrics = constants.metrics[:]

    def toggle_metric(self, metric, event):
        self.active_metrics = [metric]
        self.histogram.update(self.active_metrics)
        self.sphere.update(self.active_metrics)

    def show(self):
        for i in range(0, len(self.df), 2):
            views_metrics = df['views_metrics'][i + 1]
            metrics_2d = df.iloc[i][constants.metrics]
            metrics_3d = df.iloc[i + 1][constants.metrics]
            projection_method = df.iloc[i]['projection_name']
            self.fig = plt.figure(1)
            self.ax_3d = self.fig.add_subplot(221, projection='3d')
            self.ax_2d = self.fig.add_subplot(222)
            self.ax_sphere = self.fig.add_subplot(223, projection='3d')
            self.ax_histogram = self.fig.add_subplot(2, 2, 4)
            self.button_axes = [plt.axes([0.91, 0.05 + 0.065 * i, 0.08, 0.06]) for i in range(len(constants.metrics))]
            self.buttons = []
            self.cmap = plt.get_cmap('tab10')
            self.active_metrics = constants.metrics[:]
            self.fig.suptitle(f"D: {self.dataset_name}     P: {projection_method}")
            views_df = pd.read_pickle(f"output/{self.dataset_name}-{projection_method}-views.pkl")
            viewpoints = np.array(views_df['viewpoints'].to_list())
            for index, metric in enumerate(constants.metrics):
                self.buttons.append(Button(self.button_axes[index], metric, color=self.cmap(index, alpha=0.7),
                                           hovercolor=self.cmap(index)))
                self.buttons[-1].on_clicked(partial(self.toggle_metric, metric))

            proj_file_3d = F"output/{dataset_name}-{projection_method}-3d.csv"
            df_3d = pd.read_csv(proj_file_3d, sep=';')
            label_file = glob(F'data/{dataset_name}/*-labels.csv')
            if len(label_file) == 1:
                df_label = pd.read_csv(label_file[0], sep=';', header=0)
                labels = df_label.to_numpy()
            else:
                labels = None
            self.scatter_3d = Scatter3D(self.ax_3d, df_3d, labels)
            self.scatter_3d.draw()

            proj_file_2d = F"output/{dataset_name}-{projection_method}-2d.csv"
            df_2d = pd.read_csv(proj_file_2d, sep=';')
            self.scatter_2d = Scatter2D(self.ax_2d, df_2d, labels )
            self.scatter_2d.draw()


            self.sphere = MetricSphere(self.ax_sphere, views_metrics, viewpoints)
            self.sphere.draw()
            self.histogram = Histogram(self.ax_histogram, views_metrics, metrics_2d, metrics_3d, self.dataset_name,
                                  projection_method)
            self.histogram.draw()

            self.fig.show()
            r_scatter_previous = (0,0)
            r_sphere_previous = (0,0)
            while plt.fignum_exists(1):
                r_sphere = self.sphere.get_rotation()
                r_scatter = self.scatter_3d.get_rotation()
                if r_scatter_previous != r_sphere:
                    r_scatter_previous = r_sphere
                    self.scatter_3d.set_rotation(r_sphere[0], r_sphere[1])
                elif r_sphere_previous != r_scatter:
                    r_sphere_previous = r_scatter
                    self.sphere.set_rotation(r_scatter[0], r_scatter[1])
                plt.pause(0.01)


if __name__ == '__main__':
    dataset_name = sys.argv[1]
    metrics_file = os.path.join(constants.metrics_dir, F'metrics_{dataset_name}.pkl')
    df = pd.read_pickle(metrics_file)
    tool = Tool(df, dataset_name)
    tool.show()





