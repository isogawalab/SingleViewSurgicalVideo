import numpy as np
import argparse
import os
from scipy.spatial.transform import Rotation

def camera2world(qws, qxs, qys, qzs, txs, tys, tzs):
    q = np.stack([qws, qxs, qys, qzs], 1)
    r = Rotation.from_quat(q)
    rotation = r.as_matrix()
    translation = np.stack([txs, tys, tzs], 1)
    center_worldcoordinates = []
    for i, rt in enumerate(zip(rotation, translation)):
        center_worldcoordinates.append(-np.dot(rt[0].T, rt[1]))
    center_worldcoordinates = np.array(center_worldcoordinates)
    return center_worldcoordinates

def get_edge_len(center):
    center = center.astype(np.float32)
    edge = []
    n = center.shape[0]
    for i in range(n):
        if i ==  n - 1:
            p1 = center[i]
            p2 = center[0]
        else:
            p1 = center[i]
            p2 = center[i + 1]
        l = np.sqrt(np.sum(np.power(p1 - p2, 2)))
        edge.append(l)
    edge = np.array(edge)
    print(edge)
    return edge

def get_ave_edge(edge):
    ave = np.average(edge)
    return ave

def get_std_edge(edge):
    std = np.std(edge)
    return std

def get_camera_center(src_dir):
    # load image.txt
    qws, qxs, qys, qzs, txs, tys, tzs = np.loadtxt(os.path.join(src_dir, "assets/test/000/images.txt"), usecols=[1, 2, 3, 4, 5, 6, 7], comments='#', unpack=True)[:, ::2]
    center = camera2world(qws, qxs, qys, qzs, txs, tys, tzs)
    return center

def main(src_dir):
    # load image.txt
    qws, qxs, qys, qzs, txs, tys, tzs = np.loadtxt(os.path.join(src_dir, "assets/test/000/images.txt"), usecols=[1, 2, 3, 4, 5, 6, 7], comments='#', unpack=True)[:, ::2]
    center = camera2world(qws, qxs, qys, qzs, txs, tys, tzs)
    edge = get_edge_len(center)
    edge_std = get_std_edge(edge)   
    return edge_std




    
    
