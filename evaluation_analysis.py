import random
import time

import constants
import visualization_tool
import pyqtgraph as pg
import glob
import pickle

if __name__ == '__main__':
    #randomize evaluation set
    evaluation_set = constants.evaluation_set[1:]
    random.shuffle(evaluation_set)
    constants.evaluation_set[1:] = evaluation_set

    evaluation_files = glob.glob('evaluationdata/*.pkl')
    data = []
    for file in evaluation_files:
        with open('evaluationdata/evaluationdata.pkl', 'rb') as file:
            data.append(pickle.load(file))

    #start tool
    win = visualization_tool.Tool()
    win.set_analysis_data(data)
    pg.exec()
