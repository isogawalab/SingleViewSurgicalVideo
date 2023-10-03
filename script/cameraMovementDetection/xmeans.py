import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import cluster, preprocessing
from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
import argparse
import seaborn as sns
import os
import shutil
from statistics import mean
from scipy.signal import find_peaks

from script.cameraMovementDetection import readData


def my_xmeans3(src_dir, dir_name, data, start_frame, end_frame):
    x = data[:, 1]
    x = x.reshape([x.shape[0], 1])
    df = pd.DataFrame(data=data, columns=['x', 'y'])
    xm_c = kmeans_plusplus_initializer(x, 2).initialize()
    xm_i = xmeans(data=x, initial_centers=xm_c, kmax=2, ccore=True)
    xm_i.process()
    classes = len(xm_i._xmeans__centers)
    predict = xm_i.predict(x)
    df['predict'] = predict
    df['predict_sorted'] =  df['predict'].replace(
        df.groupby('predict')['x'].min().sort_values().index, 
        range(len(df.groupby('predict')['x'].min()))
    )
    df_deb = df.copy()
    pp = 0
    id_list = []
    drop_list = []
    for index, row in df.iterrows():
        if (pp == row.at['predict_sorted']):
            pass
        else:
            start = id_list[0]
            end = id_list[len(id_list) - 1]
            if end - start > 30:
                pass
            else:
                drop_list.extend(id_list)
            id_list = []
        id_list.append(index)
        pp = row.at['predict_sorted']
    for index in drop_list:
        df.iat[index, 3] = 100
    df = df[df['predict_sorted'] != 100]
    df['predict_sorted'] =  df['predict_sorted'].replace(
        df.groupby('predict_sorted')['x'].min().sort_values().index, 
        range(len(df.groupby('predict_sorted')['x'].min()))
    )
    x_max = df.groupby('predict_sorted')['x'].max()
    
    for index in drop_list:
        if (x_max[0] >= df_deb.iat[index, 0]):
            df_add = pd.DataFrame([[df_deb.iat[index, 0], df_deb.iat[index, 1], df_deb.iat[index, 2], 0]], columns=["x", "y", "predict", "predict_sorted"])
            df = pd.concat([df, df_add], axis=0, ignore_index=True)
        elif (x_max[0] <= df_deb.iat[index, 0]):
            df_add = pd.DataFrame([[df_deb.iat[index, 0], df_deb.iat[index, 1], df_deb.iat[index, 2], 1]], columns=["x", "y", "predict", "predict_sorted"])
            df = pd.concat([df, df_add], axis=0, ignore_index=True)
    y_mean = df.groupby('predict_sorted')['y'].mean()
    x_max = df.groupby('predict_sorted')['x'].max()
    x_mean = df.groupby('predict_sorted')['x'].mean()
    if len(y_mean) > 1:
        y_diff = abs(y_mean[0] - y_mean[1])
        x_diff = abs(x_mean[0] - x_mean[1])
        if (y_diff < 30) or ((x_diff < (end_frame - start_frame) / 3)) or (max(y_mean[0], y_mean[1]) < 50):
            df['predict_sorted'] =  df['predict_sorted'].replace(1, 0)
    ax = sns.scatterplot(x='x', y='y', hue='predict_sorted', data=df, palette='colorblind', alpha=0.5)
    centers1 = (np.array(df.groupby('predict_sorted')['x'].mean().tolist())).reshape(-1, 1)
    centers2 = (np.array(df.groupby('predict_sorted')['y'].mean().tolist())).reshape(-1, 1)
    centers = np.stack([centers1, centers2], axis=1)
    return df



def main(src_dir, dir_name, start_frame, end_frame):
    src_path = os.path.join(src_dir, 'txt/split', str(dir_name), 'detectMovement_data_smooth.txt')
    data = np.array(readData.read_data2(src_path))
    df = my_xmeans3(src_dir, dir_name, data, start_frame, end_frame)
    peaks, _ = find_peaks(df['y'])
    class0_max_peak = 0
    class1_max_peak = 0
    for p in peaks:
        if (df['predict_sorted'][p] == 0):
            if (class0_max_peak < df['y'][p]):
                class0_max_peak = df['y'][p]
        if (df['predict_sorted'][p] == 1):
            if (class1_max_peak < df['y'][p]):
                class1_max_peak = df['y'][p]
    y_max = df.groupby('predict_sorted')['y'].max()
    y_min = df.groupby('predict_sorted')['y'].min()
    y_mean = df.groupby('predict_sorted')['y'].mean()
    if len(y_max) == 1:
        th = 0
        diff = 0
    else:
        th = min(min(y_max[0], y_max[1]) + 1, min(y_mean[0], y_mean[1]) * 2)
        diff = abs(y_mean[0] - y_mean[1])
        if diff < 100:
            th = min(y_max[0], y_max[1]) + 1
        if th < min(class0_max_peak, class1_max_peak):
            th = min(min(class0_max_peak, class1_max_peak) + 10, min(y_max[0], y_max[1]) + 1)
        
    return th

