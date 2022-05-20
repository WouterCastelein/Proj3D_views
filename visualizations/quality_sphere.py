from pyqtgraph.opengl import shaders
from pyqtgraph.opengl.shaders import ShaderProgram, VertexShader, FragmentShader

from visualizations.synced_camera_view_widget import SyncedCameraViewWidget
import constants
import numpy as np
import pyqtgraph.opengl as gl
from scipy.spatial import ConvexHull


class CustomMeshItem(gl.GLMeshItem):
    def __init__(self, texture_map, *args, **kwargs):
        super(CustomMeshItem, self).__init__(*args, **kwargs)

        #Initalize shader that used 1D texture to map metric data to colors, and adds simple shading
        self.custom_shader = ShaderProgram('custom_shader', [
            VertexShader("""
                varying vec3 normal;
                void main() {
                    // compute here for use in fragment shader
                    normal = normalize(gl_NormalMatrix * gl_Normal);
                    gl_FrontColor = gl_Color;
                    gl_BackColor = gl_Color;
                    gl_Position = ftransform();
                }
            """),
            FragmentShader("""
                varying vec3 normal;
                uniform float colorMap[20];
                void main() {
                    float p = dot(normal, normalize(vec3(0.01, 0.01, -0.50)));
                    p = p < 0. ? 0. : p;
                    float m_value = gl_Color.x;
                    int i = 0;
                    float thresholds[5] = float[5](0., 0.2, 0.5, 0.8, 1.0);
                    for(int j = 1;j<5;j++){
                        if (m_value > thresholds[j])
                            i++;
                    }
                    float ratio = (m_value - thresholds[i]) / (thresholds[i + 1] - thresholds[i]);
                    i = i * 4;
                    vec3 color1 = vec3(colorMap[0 + i], colorMap[1 + i], colorMap[2 + i]) * (1.0 - ratio);
                    vec3 color2 = vec3(colorMap[4 + i], colorMap[5 + i], colorMap[6 + i]) * ratio;
                    vec4 color = vec4(color1 + color2, 1.0);
                    
                    //shading
                    color.x = color.x * p;
                    color.y = color.y * p;
                    color.z = color.z * p;
                    gl_FragColor = color;
                }
            """)
        ], uniforms={'colorMap': texture_map}),

class QualitySphere(SyncedCameraViewWidget):
    def __init__(self, data, cmap, parent=None, *args, **kwargs, ):
        super().__init__(*args, **kwargs)
        self.data = data
        self.parent = parent
        self.setCameraPosition(distance=5)
        vertices = np.load(f'spheres/sphere{constants.samples}_points.npy')
        faces = np.load(f'spheres/sphere{constants.samples}_faces.npy')
        self.cmap = cmap

        #Store the metric data into the vertex colors
        vertex_colors = [(x, x, x, 1) for x in self.data]

        #Generate 1D texture to map metric data to actual colors:
        texture = self.cmap.mapToFloat([0, 0.2, 0.5, 0.8, 1.0])
        texture_1d = [color_value for color in texture for color_value in color]

        self.md = gl.MeshData(vertexes=vertices, faces=faces, vertexColors=vertex_colors)
        self.mesh_item = CustomMeshItem(texture_1d, meshdata=self.md, smooth=True, shader='custom_shader', glOptions='translucent')
        self.addItem(self.mesh_item)

        #Add white dot on the current viewpoint:
        eye = self.cameraPosition()
        eye.normalize()
        self.scatter_item = gl.GLScatterPlotItem(pos=eye, size=5, color=(1,1,1, 1),
                                                 pxMode=True)
        self.scatter_item.setGLOptions('translucent')
        self.addItem(self.scatter_item)

    def on_view_change(self):
        super(QualitySphere, self).on_view_change()
        self.draw_crosshair()

    def draw_crosshair(self):
        eye = self.cameraPosition()
        eye.normalize()
        self.scatter_item.setData(pos=eye)