from sklearn.ensemble import IsolationForest
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

from script.cameraMovementDetection import readData


def rm_outlier(src_dir, data, dir_name):
    clf = IsolationForest(n_estimators=100)
    clf.fit(data)
    predict = clf.decision_function(data)
    threshold = -0.1
    predict_normal = data[np.intersect1d((np.where(predict > threshold)), (np.where(data[:, 1] != 0)))]
    predict_abnormal = data[np.where(predict <= threshold)]
    predict_0 = data[np.where(data[:, 1] == 0)]
    return predict_normal




