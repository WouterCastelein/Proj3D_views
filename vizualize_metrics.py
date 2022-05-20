import pandas as pd
import sys
from glob import glob

import visualization_tool
import pyqtgraph as pg

if __name__ == '__main__':
    #dataset_name = sys.argv[1]
    win = visualization_tool.Tool()
    pg.exec()