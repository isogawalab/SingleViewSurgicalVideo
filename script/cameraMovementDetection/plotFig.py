import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d # scipyのモジュールを使う
import sys
import argparse
import os
import shutil
import math

from script.cameraMovementDetection import readData
from script.cameraMovementDetection import rmOutlier
from script.cameraMovementDetection import xmeans

FPS = 30
M_AVE = 10
DEBUG = 1

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

def valid_convolve(xx, size):
    b = np.ones(size)/size
    xx_mean = np.convolve(xx, b, mode="same")
    n_conv = math.ceil(size/2)
    xx_mean[0] *= size/n_conv
    for i in range(1, n_conv):
        xx_mean[i] *= size/(i+n_conv)
        xx_mean[-i] *= size/(i + n_conv - (size % 2))
    return xx_mean

def moving_avg(in_data):
    in_x = in_data[:, 0]
    in_y = in_data[:, 1]
    np_y_conv = valid_convolve(in_y, M_AVE)
    out_x_dat = in_x
    out_data = np.stack([out_x_dat, np_y_conv], axis=1)
    return out_data
    
def main2(src_dir, th):
    src_path = os.path.join(src_dir, 'txt/detectMovement_data.txt')
    bn = os.path.basename(src_path)
    data = np.array(readData.read_data(src_path))
    data0 = rmOutlier.rm_outlier(src_dir, data, 0)
    d = []
    for i in range(int(data0[0, 0]), int(data0[data0.shape[0] - 1, 0]), FPS):
        dd = interpolated_intercept(data0[:, 1], data0[:, 0], np.array([i] * data0.shape[0]))
        d.append([dd[1][0][0], dd[0][0][0]])
    d = np.array(d)
    data1 = d
    data2 = spline_interp(data1)
    data4 = moving_avg(data0)
    xc, yc = interpolated_intercept(data4[:, 0], data4[:, 1], np.array([th] * data4.shape[0]))
    point_list = []
    for i, point in enumerate(zip(xc, yc)):
        point_list.append(point[0][0])

    return point_list



def main(src_dir, dir_name, start_frame, end_frame, th):
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
    point_list = []
    if th == 0:
        pass
    else:
        xc, yc = interpolated_intercept(data4[:, 0], data4[:, 1], np.array([th] * data4.shape[0]))
        if (xc.shape[0] < 10):
            for i, point in enumerate(zip(xc, yc)):
                point_list.append(point[0][0])

    # あとで消すここから
    plt.xlim(start_frame, end_frame)
    if os.path.isfile(os.path.join(src_dir, 'fig/split', str(dir_name), 'detectMovement3_data_smooth.png')):
        for j in range(0, 1000):
            if os.path.isfile(os.path.join(src_dir, 'fig/split', str(dir_name), 'detectMovement3_data_smooth_' + str(j) + '.png')):
                continue
            else:
                shutil.copyfile(os.path.join(src_dir, 'fig/split', str(dir_name), 'detectMovement3_data_smooth.png'), os.path.join(src_dir, 'fig/split', str(dir_name), 'detectMovement3_data_smooth_' + str(j) + '.png'))
                break
    plt.plot(data1[:, 0], data1[:, 1], color='b', label='original', alpha=0.3)
    plt.xlabel('frame')
    plt.ylabel('amount of deviation')
    plt.legend(frameon=False)
    plt.savefig(os.path.join(src_dir, 'fig/split', str(dir_name), str(bn)[0:len(str(bn)) - 4] + '_smooth_0.png'))
    # plt.plot(data2[:, 0], data2[:, 1], color='r', label='spline', alpha=0.7)
    plt.plot(data4[:, 0], data4[:, 1], color='g', label='smooth', alpha=1.0)
    plt.xlabel('frame')
    plt.ylabel('amount of deviation')
    plt.legend(frameon=False)
    plt.savefig(os.path.join(src_dir, 'fig/split', str(dir_name), str(bn)[0:len(str(bn)) - 4] + '_smooth_1.png'))
    point_list = []
    if th == 0:
        pass
    else:
        plt.hlines([th], data1[0, 0], data1[data1.shape[0] - 1, 0], "red", linestyles='dashed', label="threshold")
        xc, yc = interpolated_intercept(data4[:, 0], data4[:, 1], np.array([th] * data4.shape[0]))
        if (xc.shape[0] < 10):
            plt.plot(xc[:, 0], yc[:, 0], 'ro', ms=5, label='point of intersection')
            for i, point in enumerate(zip(xc, yc)):
                plt.text(point[0][0], point[1][0], '{x}'.format(x=round(point[0][0], 0)), fontsize=10)
                point_list.append(point[0][0])
            plt.legend(frameon=False)

    plt.xlabel('frame')
    plt.ylabel('amount of deviation')
    plt.savefig(os.path.join(src_dir, 'fig/split', str(dir_name), str(bn)[0:len(str(bn)) - 4] + '_smooth.png'))
    plt.clf()

    # あとで消すここまで

    return point_list
    
