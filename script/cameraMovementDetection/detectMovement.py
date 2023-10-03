import numpy as np
import cv2
import os
import sys
import argparse
from natsort import natsorted
from matplotlib import pyplot as plt
import glob
from statistics import stdev
import pandas as pd
import shutil
from tqdm import tqdm


"""
setting of params
"""
SKIP_NUM = 30
TH = 100
MAX = 500


def main(src_dir):
    videos = natsorted(glob.glob(os.path.join(src_dir, 'videos/output/*')))

    caps = []
    for i, video in enumerate(videos):
        cap = cv2.VideoCapture(video)
        caps.append(cap)
        if not cap.isOpened():
            sys.exit()
    
    flag = 0
    count = 0
    data = []
    x = []
    over_th = []

    with tqdm() as pbar:
        while True:
            debnum = 0
            frame_list = []
            for i in range(len(caps)):
                ret, frame = caps[i].read()
                if not ret:
                    flag = 1
                    break
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame_list.append(frame_gray)
            if flag:
                break

            if not(count % SKIP_NUM == 0):
                count += 1
                continue
            else:
                count += 1

            combination_num = 0
            sum_all = 0
            sift = cv2.SIFT_create()
            for i in range(len(caps)):
                frame1 = frame_list[i]
                kp1, des1 = sift.detectAndCompute(frame1,None)
                if (len(kp1) == 0):
                    continue
                for j in range(i + 1, len(caps)):  # frame2
                    frame2 = frame_list[j]
                    kp2, des2 = sift.detectAndCompute(frame2,None)
                    if (len(kp2) == 0):
                        continue
                    bf = cv2.BFMatcher()
                    matches = bf.knnMatch(des1,des2, k=2)
                    good = []
                    try:
                        for m,n in matches:
                            if m.distance < 0.75*n.distance:
                                good.append([m])
                    except ValueError:
                        continue
                    if len(good) < 10:
                        continue
                    pts1 = []
                    pts2 = []
                    for k,m in enumerate(good):
                        pts1.append(kp1[m[0].queryIdx].pt)
                        pts2.append(kp2[m[0].trainIdx].pt)
                    pts1 = np.int32(pts1)
                    pts2 = np.int32(pts2)
                    M, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC,5.0)
                    pts1_inliers = pts1[mask.ravel() == 1]
                    pts2_inliers = pts2[mask.ravel() == 1]
                    
                    sum_err = 0
                    std_list = []
                    for p1, p2 in zip(pts1_inliers, pts2_inliers):
                        err_vec = abs(p1 - p2)
                        err = np.linalg.norm(err_vec)
                        sum_err += err
                        std_list.append(err)
                    if (len(pts1_inliers)  == 0):
                        continue
                    sum_err /= len(pts1_inliers)
                    std = stdev(std_list)
                    if (std > 100):
                        continue
                    combination_num += 1
                    sum_all += sum_err

            if combination_num == 0:
                result = 0
            else:
                result = sum_all / combination_num
            if result < TH and result != 0:
                over_th.append(count - 1)
            data.append(result)
            x.append(count - 1)           
            pbar.update(1)

    with open(os.path.join(src_dir, 'txt/detectMovement_data.txt'), 'w') as f1:
        for i, d in enumerate(data):
            print(d, file=f1)
    
    return 

