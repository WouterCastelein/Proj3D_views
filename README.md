# Thesis
Master thesis on evaluating 3D projections by its 2D views instead of the entire 3D projection at once.

# proj3d

Python version 3.9.9
Install packages: 'pip install -r requirements.txt'

#Troubleshooting

**Cannot find/open MulticoreTSNE shared library:** \
MulticoreTSNE package might not work on 64bit, in that case \
-If you installed package before, pip uninstall MulticoreTSNE \
-Clone the source: git clone https://github.com/DmitryUlyanov/Multicore-TSNE.git \
-Edit /Multicore-TSNE/setup.py

Add the following bold line to setup.py:

if 0 != execute(['cmake', 
**'-DCMAKE_GENERATOR_PLATFORM=x64',**
'-DCMAKE_BUILD_TYPE={}'.format(build_type),
'-DCMAKE_VERBOSE_MAKEFILE={}'.format(int(self.verbose)),

next install the package from the local repository (pip install .) or if you use a 
virtual environment make sure you install it there.

**Tapkee executable not found:**
Go to: https://github.com/lisitsyn/tapkee/releases/tag/1.2
Download the tapkee executable for your operating system, and replace the file tapkee/tapkee
with the executable renamed to 'tapkee.exe'