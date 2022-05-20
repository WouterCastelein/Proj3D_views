"""
Simple example using BarGraphItem
"""
import math
from glob import glob
import os
import numpy as np
import pandas as pd

import pyqtgraph as pg
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtGui import QColor
from pyqtgraph import mkPen

import utils
import constants
from visualizations.scatter_3d import Scatter3D
from visualizations.scatter_2d import Scatter2D
from visualizations.quality_sphere import QualitySphere
from visualizations.parallel_bar_plot import parallelBarPlot

class Tool(pg.GraphicsWindow):
    def __init__(self, dataset_name="Concrete", projection_method="TSNE"):
        super(Tool, self).__init__()
        self.dataset_name = dataset_name
        self.projection_method = projection_method
        metrics_file = os.path.join(constants.metrics_dir, f'metrics_{dataset_name}.pkl')
        df = pd.read_pickle(metrics_file)
        select = df.loc[df['projection_name'] == projection_method]

        self.views_metrics = select.iloc[1]['views_metrics'][:, 1:]
        self.metrics_2d = select.iloc[0][constants.metrics].to_numpy()
        self.metrics_3d = select.iloc[1][constants.metrics].to_numpy()
        self.move_to_view_info = None

        self.setBackground((0, 0, 0, 60))
        self.labels = self.get_labels()

        self.layoutgb = QtWidgets.QGridLayout()
        self.layoutgb.setHorizontalSpacing(1)
        self.layoutgb.setVerticalSpacing(1)
        self.layoutgb.setContentsMargins(1,1,1,1)

        self.setLayout(self.layoutgb)

        self.view_points = np.load(f'spheres/sphere{constants.samples}_points.npy')

        self.current_metric = 'normalized_stress'
        self.D_P_dict = self.available_datasets_projections()

        self.initialize_3d_scatterplot()
        self.initialize_2d_scatterplot()
        self.initialize_sphere()
        self.initialize_histogram()

        self.ci.layout.setContentsMargins(0,0,0,0)

        self.layoutgb.setColumnMinimumWidth(0, 50)
        self.layoutgb.setColumnStretch(1, 20)
        self.layoutgb.setColumnStretch(2, 20)
        self.layoutgb.setRowStretch(0, 10)
        self.layoutgb.setRowStretch(1, 10)

        self.initialize_menu()
        self.highlight()


    def get_labels(self):
        label_file = glob(F'data/{self.dataset_name}/*-labels.csv')
        if len(label_file) == 1:
            df_label = pd.read_csv(label_file[0], sep=';', header=0)
            return df_label.to_numpy().flatten()
        else:
            return None

    def available_datasets_projections(self):
        consolid_metrics = os.path.join(constants.metrics_dir, 'metrics.pkl')
        data_frame = pd.read_pickle(consolid_metrics)
        D_P_dict = {}
        datasets = set(data_frame['dataset_name'].to_list())
        for dataset in datasets:
            D_P_dict[dataset.split('.')[0]] = set(data_frame[data_frame['dataset_name'] == dataset]['projection_name'].to_list())
        return D_P_dict

    def initialize_menu(self):
        keys = list(self.D_P_dict.keys())
        self.dataset_picker = pg.ComboBox(items=list(self.D_P_dict.keys()), default=self.dataset_name)
        self.proj_technique_picker = pg.ComboBox(items=list(self.D_P_dict[keys[0]]), default=self.projection_method)
        self.sphere_metric_picker = pg.ComboBox(items=constants.metrics, default=self.current_metric)

        self.dataset_picker.currentIndexChanged.connect(self.data_selected)
        self.proj_technique_picker.currentIndexChanged.connect(self.data_selected)
        self.sphere_metric_picker.currentIndexChanged.connect(self.metric_selected)

        self.menu = pg.LayoutWidget()

        #Set background white:
        palette = QtGui.QPalette()
        palette.setColor(QtGui.QPalette.Window, QColor(255,255,255,255))
        self.menu.setPalette(palette)
        self.menu.setAutoFillBackground(True);


        self.menu.addLabel(text="Dataset:", row=0, col=0)
        self.menu.addWidget(self.dataset_picker, 1, 0)
        self.menu.addLabel(text="Projection Method:", row=2, col=0)
        self.menu.addWidget(self.proj_technique_picker, 3, 0)
        self.menu.addLabel(text="Sphere metric:", row=4, col=0)
        self.menu.addWidget(self.sphere_metric_picker, row=5, col=0)
        self.menu.layout.setRowStretch(0, 0)
        self.menu.layout.setRowStretch(1, 0)
        self.menu.layout.setRowStretch(2, 0)
        self.menu.layout.setRowStretch(3, 0)
        self.menu.layout.setRowStretch(4, 0)
        self.menu.layout.setRowStretch(5, 0)
        self.menu.layout.setRowStretch(6, 1)
        self.layoutgb.addWidget(self.menu, 0, 0, 3, 1)

    def initialize_3d_scatterplot(self):
        proj_file_3d = F"{constants.output_dir}/{self.dataset_name}-{self.projection_method}-3d.csv"
        data = pd.read_csv(proj_file_3d, sep=';').to_numpy()
        self.scatter_3d = Scatter3D(data, self.labels, parent=self, title="3D projection")
        self.scatter_3d.setBackgroundColor('w')
        self.layoutgb.addWidget(self.scatter_3d, 0, 1)

    def initialize_2d_scatterplot(self):
        # 2D Scatter
        proj_file_2d = F"{constants.output_dir}/{self.dataset_name}-{self.projection_method}-2d.csv"
        df_2d = pd.read_csv(proj_file_2d, sep=';').to_numpy()
        self.scatter_2d = Scatter2D(df_2d, self.labels)
        self.layoutgb.addWidget(self.scatter_2d, 0, 2)
        self.scatter_2d.setBackground('w')

    def initialize_sphere(self):
        self.sphere_data = np.copy(self.views_metrics[:, constants.metrics.index(self.current_metric)])

        c = ["darkred", "red", "yellow", "green", "darkgreen"]
        v = [ 0, 0.2, 0.5, 0.8, 1]
        self.cmap = pg.ColorMap(v, c)

        self.sphere = QualitySphere(self.sphere_data, self.cmap, parent=self, title=F"Viewpoint quality ({self.current_metric})")
        self.sphere.setBackgroundColor('w')

        self.sphere_widget = pg.LayoutWidget()
        self.sphere_widget.addWidget(self.sphere, 0, 0)
        self.sphere_widget.layout.setContentsMargins(0, 0, 0, 0)
        self.sphere_widget.layout.setHorizontalSpacing(0)

        self.cbw = pg.GraphicsLayoutWidget()
        self.color_bar = pg.ColorBarItem(colorMap=self.cmap, interactive=False, values=(0, 1))
        self.color_bar_line = self.color_bar.addLine(y=255, pen=mkPen(0,0,0,255))
        self.cbw.addItem(self.color_bar)
        self.cbw.setBackground('w')
        self.sphere_widget.addWidget(self.cbw, 0, 1)
        self.sphere_widget.layout.setColumnStretch(1, 2)
        self.sphere_widget.layout.setColumnStretch(0, 8)

        self.cbw.setSizePolicy(self.sphere.sizePolicy())
        self.layoutgb.addWidget(self.sphere_widget, 1, 1)

        self.scatter_3d.sync_camera_with(self.sphere)
        self.sphere.sync_camera_with(self.scatter_3d)

    def initialize_histogram(self):
        self.hist = parallelBarPlot(self.views_metrics, self.metrics_2d, self.metrics_3d, self.view_points, parent=self)
        self.hist.setBackground('w')
        self.layoutgb.addWidget(self.hist, 1, 2)

    def highlight(self):
        eye = self.sphere.cameraPosition()
        eye.normalize()

        #Find the viewpoint for which me have metrics. that is closest to the current viewpoint
        distances = np.sum((self.view_points - np.array(eye))**2, axis=1)
        nearest = np.argmin(distances)

        #Get the metric values, and highlight the corresponding histogram bars
        nearest_values = self.views_metrics[nearest]
        self.hist.highlight_bar_with_values(nearest_values)

        #Update the line in the sphere colorbar
        metric_score = nearest_values[constants.metrics.index(self.current_metric)]
        self.color_bar_line.setValue(255 * metric_score)

    def move_to_view(self, metric_index, bin_index, metric_value_l, metric_value_r, percentage):
        a = self.views_metrics[:, metric_index]
        indices = np.argwhere(np.logical_and(a >= metric_value_l, a <= metric_value_r)).flatten()
        indices = indices[np.argsort(self.views_metrics[indices, metric_index])]
        index = indices[round((len(indices) - 1) * percentage)]
        viewpoint = utils.rectangular_to_spherical(np.array([self.view_points[index]]))[0]
        self.sphere.setCameraPosition(azimuth=viewpoint[1], elevation=viewpoint[0], distance=self.sphere.cameraParams()['distance'])
        self.scatter_3d.setCameraPosition(azimuth=viewpoint[1], elevation=viewpoint[0], distance=self.scatter_3d.cameraParams()['distance'])
        self.sphere.update_views()
        self.scatter_3d.update_order()

    def data_selected(self):
        dataset_name = self.dataset_picker.value()
        projection_method = self.proj_technique_picker.value()
        self.set_data(dataset_name, projection_method)
        pass

    def metric_selected(self):
        self.current_metric = self.sphere_metric_picker.value()
        self.initialize_sphere()
        self.scatter_3d.update_views()

    def set_data(self, dataset_name, projection_method):
        """
        Update the data of all the widgets inside the tool to a new dataset and projection technique combination
        """
        self.dataset_name = dataset_name
        self.projection_method = projection_method
        metrics_file = os.path.join(constants.metrics_dir, F'metrics_{self.dataset_name}.pkl')
        df = pd.read_pickle(metrics_file)
        select = df.loc[df['projection_name'] == self.projection_method]

        self.views_metrics = select.iloc[1]['views_metrics'][:, 1:]
        self.metrics_2d = select.iloc[0][constants.metrics].to_numpy()
        self.metrics_3d = select.iloc[1][constants.metrics].to_numpy()
        self.initialize_histogram()
        self.initialize_sphere()


        self.labels = self.get_labels()
        proj_file_3d = F"{constants.output_dir}/{self.dataset_name}-{self.projection_method}-3d.csv"
        data3d = pd.read_csv(proj_file_3d, sep=';').to_numpy()
        self.scatter_3d.set_data(data3d, self.labels)


        proj_file_2d = F"{constants.output_dir}/{self.dataset_name}-{self.projection_method}-2d.csv"
        data2d = pd.read_csv(proj_file_2d, sep=';').to_numpy()
        self.scatter_2d.set_data(data2d, self.labels)














