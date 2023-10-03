import numpy as np
import cv2
import os
import sys
import argparse
from natsort import natsorted
from matplotlib import pyplot as plt
from tqdm import tqdm
import glob

SKIP_NUM = 30
TIME = 300
TH = 0.5
TH2 = 1.0
TH3 = 1.5
MAX = 2

HSV_MIN1 = np.array([0, 0, 10])
HSV_MAX1 = np.array([30, 255, 255])
HSV_MIN2 = np.array([150, 0, 10])
HSV_MAX2 = np.array([179, 255, 255])

HSV_MIN3 = np.array([0, 0, 240])
HSV_MAX3 = np.array([179, 10, 255])




def image_hcombine(frames):
    for j, frame in enumerate(frames):
        frame = cv2.resize(frame,(200, 200))
        if j != 0:
            im_tile = cv2.hconcat([im_tile, frame])
        else:
            im_tile = frame
    return im_tile



def make_oktime_list(over_th):
    oktime = []
    count = 0
    prev = 0
    for i, over in enumerate(over_th):
        if (over == 0 or over == prev + SKIP_NUM):
            if (count == 0):
                start = prev
            count += 1
            s = 1
        elif (over < prev + SKIP_NUM * 3):
            count = count
            s = 2
        else:
            if (count * SKIP_NUM > TIME): 
                end = prev
                oktime.append([start, end])
            count = 0
            s = 3
        prev = over
    if (s == 1):
        end = prev
        oktime.append([start, end])
    return oktime


def main(src_dir):
    videos = natsorted(glob.glob(os.path.join(src_dir, 'videos/input/*')))
    caps = []
    for i, video in enumerate(videos):
        cap = cv2.VideoCapture(video)
        caps.append(cap)
        if not cap.isOpened():
            sys.exit()
    
    flag = 0
    count = 0
    data = []
    data_ave = []
    x = []
    over_th = []
    over_th2 = []
    over_th3 = []
    with tqdm() as pbar:
        while True:
            frame_list = []
            for i in range(len(caps)):
                ret, frame = caps[i].read()
                if not ret:
                    flag = 1
                    break
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame_list.append(frame)
            if flag:
                break
            if not(count % SKIP_NUM == 0):
                count += 1
                continue
            else:
                count += 1
            sum = 0
            sum_list = []
            flag_shirotobi = 0
            for i in range(len(caps)):
                frame1 = frame_list[i]
                h, w, _ = frame1.shape
                hsv = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
                mask_hsv1 = cv2.inRange(hsv, HSV_MIN1, HSV_MAX1)
                mask_hsv2 = cv2.inRange(hsv, HSV_MIN2, HSV_MAX2)
                mask = np.bitwise_or(mask_hsv1, mask_hsv2)
                whitePixels = cv2.countNonZero(mask)
                mask_hsv3 = cv2.inRange(hsv, HSV_MIN3, HSV_MAX3)
                whitePixels3 = cv2.countNonZero(mask_hsv3)
                percent = whitePixels3 / (h * w)
                if (percent > 0.1):
                    flag_shirotobi += 1
                sum += whitePixels
                sum_list.append(whitePixels)
            ave = sum / len(caps)
            sum_list.sort()
            if (ave < 500):
                result = MAX
            else:
                result = (sum_list[len(sum_list) - 1] - sum_list[0]) / ave
            if result < TH and result != MAX and flag_shirotobi == 0:
                over_th.append(count - 1)
            if result < TH2 and result != MAX:
                over_th2.append(count - 1)
            if result < TH3 and result != MAX:
                over_th3.append(count - 1)
            data.append(result)
            data_ave.append(ave)
            x.append(count - 1)
            pbar.update(1)

    with open(os.path.join(src_dir, 'txt/detectSurgicalField_data.txt'), 'w') as f1:
        for i, d in enumerate(data):
            print(d, file=f1)
    oktime = make_oktime_list(over_th)
    oktime2 = make_oktime_list(over_th2)
    oktime3 = make_oktime_list(over_th3)
    with open(os.path.join(src_dir, 'txt/detectSurgicalField_time_0.txt'), 'w') as f2:
        for i, d in enumerate(oktime):
            # f.write("%s\n" % d)
            print(d[0], end=" ", file=f2)
            print(d[1], end="\n", file=f2)
    with open(os.path.join(src_dir, 'txt/detectSurgicalField_time_1.txt'), 'w') as f2:
        for i, d in enumerate(oktime2):
            # f.write("%s\n" % d)
            print(d[0], end=" ", file=f2)
            print(d[1], end="\n", file=f2)
    with open(os.path.join(src_dir, 'txt/detectSurgicalField_time_2.txt'), 'w') as f2:
        for i, d in enumerate(oktime3):
            # f.write("%s\n" % d)
            print(d[0], end=" ", file=f2)
            print(d[1], end="\n", file=f2)

    return

