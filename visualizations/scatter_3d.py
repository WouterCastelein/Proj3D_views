from pyqtgraph.Qt import QtGui
from pyqtgraph.opengl import shaders
from pyqtgraph.opengl.shaders import ShaderProgram, VertexShader, FragmentShader

from visualizations.synced_camera_view_widget import SyncedCameraViewWidget
import pyqtgraph.opengl as gl
from matplotlib import cm
import numpy as np
import pyqtgraph as pg
from OpenGL.GL import *

class CustomScatterItem(gl.GLScatterPlotItem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def initializeGL(self):
        super(CustomScatterItem, self).initializeGL()

        #Custom shader draws dark edges around points
        self.shader = ShaderProgram('pointSprite', [
            VertexShader("""
                                void main() {
                                    gl_PointSize = gl_Normal.x / 1.2;
                                    gl_Position = ftransform();
                                    gl_FrontColor = gl_Color; 
                                } 
                            """),
            FragmentShader("""
                            #version 120
                            uniform sampler2D texture;
                            void main ( )
                            {
                            float dist = sqrt(pow(gl_PointCoord.x - 0.5, 2) + pow(gl_PointCoord.y - 0.5, 2));
                            if (dist >= 0.30 && dist < 0.50)
                                {
                                    float diff = 0.05 * (0.2 / (0.50 - dist));
                                    gl_FragColor = texture2D(texture, gl_PointCoord) * gl_Color - vec4(diff,diff,diff,0);
                                }
                            else if (dist < 0.30)
                                gl_FragColor = texture2D(texture, gl_PointCoord) * gl_Color;
                            else
                                gl_FragColor = vec4(0,0,0,0);
                            }
                    """)
        ])

class Scatter3D(SyncedCameraViewWidget):
    def __init__(self, data, labels, parent=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = data - (np.max(data) - np.min(data)) / 2 #Center the data around (0,0,0)
        self.labels = labels
        self.parent = parent
        self.cmap = cm.get_cmap('tab10')
        self.setCameraPosition(distance=1.8)
        self.color = np.empty((data.shape[0], 4))
        if labels is None:
            for i in range(data.shape[0]):
                self.color[i] = self.cmap(0)
        else:
            for i in range(data.shape[0]):
                self.color[i] = self.cmap(self.labels[i])
        sorted_indices = self.sorted_indices()
        self.scatter_item = CustomScatterItem(pos=data[sorted_indices], size=7, color=self.color[sorted_indices], pxMode=True)
        self.scatter_item.setGLOptions('translucent')

        self.addItem(self.scatter_item)
        self.update_order()

    def update_order(self):
        """
        When drawing in translucent mode, items need to be drawn in order from back to front, for proper clipping,
        this function recomputes the point order and changes the point color based on distance from the eye.
        """
        eye = self.cameraPosition()
        distances = (-np.linalg.norm(self.data - eye, axis=1))
        distances -= np.min(distances)
        distances /= np.max(distances)
        distances -= 1.0
        sorted_indices = np.argsort(distances)
        full = np.full((distances.shape[0], 4), np.array([0.35, 0.35, 0.35, 0]))
        color_adjustment = np.multiply(full, -distances[:, None])
        self.scatter_item.setData(pos=self.data[sorted_indices], size=7, color=self.color[sorted_indices] + color_adjustment[sorted_indices])

    def sorted_indices(self):
        """
        Get the indices of the data sorted by distance from the camera position,
        used for rendering closer points on top of further points
        """
        eye = self.cameraPosition()
        return np.argsort(-np.linalg.norm(self.data - eye, axis=1))

    def on_view_change(self):
        super().on_view_change()
        self.update_order()
        self.parent.highlight()

    def set_data(self, data, labels):
        self.data = data - (np.max(data) - np.min(data)) / 2 #Center the data around (0,0,0)
        self.color = np.empty((self.data.shape[0], 4))
        self.labels = labels
        if labels is None:
            for i in range(self.data.shape[0]):
                self.color[i] = self.cmap(0)
        else:
            for i in range(self.data.shape[0]):
                self.color[i] = self.cmap(self.labels[i])
        sorted_indices = self.sorted_indices()
        self.scatter_item.setData(pos=self.data[sorted_indices], color=self.color[sorted_indices])
        self.on_view_change()
        self.update_views()

