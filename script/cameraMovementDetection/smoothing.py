import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d # scipyのモジュールを使う
import scipy.signal
import sys
import argparse
import os

from script.cameraMovementDetection import readData
from script.cameraMovementDetection import rmOutlier

TH = 75
FPS = 30
M_AVE = 10

def interpolated_intercept(x, y1, y2):
    def intercept(point1, point2, point3, point4):
        def line(p1, p2):
            A = (p1[1] - p2[1])
            B = (p2[0] - p1[0])
            C = (p1[0]*p2[1] - p2[0]*p1[1])
            return A, B, -C
        def intersection(L1, L2):
            D  = L1[0] * L2[1] - L1[1] * L2[0]
            Dx = L1[2] * L2[1] - L1[1] * L2[2]
            Dy = L1[0] * L2[2] - L1[2] * L2[0]
            x = Dx / D
            y = Dy / D
            return x,y
        L1 = line([point1[0],point1[1]], [point2[0],point2[1]])
        L2 = line([point3[0],point3[1]], [point4[0],point4[1]])
        R = intersection(L1, L2)
        return R
    idx = np.argwhere(np.diff(np.sign(y1 - y2)) != 0)
    xc, yc = intercept((x[idx], y1[idx]),((x[idx+1], y1[idx+1])), ((x[idx], y2[idx])), ((x[idx+1], y2[idx+1])))
    return xc,yc


def spline_interp(in_data):
    in_x = in_data[:, 0]
    in_y = in_data[:, 1]
    out_x = np.linspace(np.min(in_x), np.max(in_x), np.size(in_x)*100)
    func_spline = interp1d(in_x, in_y, kind='cubic')
    out_y = func_spline(out_x)
    out_data = np.stack([out_x, out_y], axis=1)
    return out_data

def moving_avg(in_data):
    in_x = in_data[:, 0]
    in_y = in_data[:, 1]
    np_y_conv = np.convolve(in_y, np.ones(M_AVE)/float(M_AVE), mode='valid')
    out_x_dat = np.linspace(np.min(in_x), np.max(in_x), np.size(np_y_conv))
    out_data = np.stack([out_x_dat, np_y_conv], axis=1)
    return out_data
    
    

def main(src_dir, dir_name, start_frame, end_frame):
    src_path = os.path.join(src_dir, 'txt/split', str(dir_name), 'detectMovement_data.txt')
    bn = os.path.basename(src_path)
    data = np.array(readData.read_data2(src_path))
    data0 = rmOutlier.rm_outlier(src_dir, data, dir_name)
    d = []
    for i in range(int(data0[0, 0]), int(data0[data0.shape[0] - 1, 0]), FPS):
        dd = interpolated_intercept(data0[:, 1], data0[:, 0], np.array([i] * data0.shape[0]))
        d.append([dd[1][0][0], dd[0][0][0]])
    d = np.array(d)
    data1 = d
    data4 = moving_avg(data0)
    data4 = data4[np.where(data4[:, 0] >= start_frame) and np.where(data4[:, 0] < end_frame)]
    data5 = []
    flag = 0
    for d in data:
        dd = interpolated_intercept(data4[:, 1], data4[:, 0], np.array([d[0]] * data4.shape[0]))
        if dd[1].shape[0] == 0 and len(data5) != 0:
            data5.append([d[0], data5[len(data5) - 1][1]])
        elif dd[1].shape[0] == 0:
            data5.append([d[0], 0])
        else:
            if flag == 0:
                data5 = [[m[0], dd[0][0][0]] if m[1] == 0 else m for m in data5]
                flag = 1
            data5.append([dd[1][0][0], dd[0][0][0]])
    data5 = np.array(data5)
    with open(src_path[0:len(src_path) - 4] + '_smooth.txt', 'w') as f1:
        for i, d in enumerate(data5):
            print(str(d[0]) + " " + str(d[1]), file=f1)
    return
    

