import os
import sys
import cv2
import glob
import shutil
import pathlib
import argparse
import itertools
import subprocess
import numpy as np
import sympy as sp
import pandas as pd
from tqdm import tqdm
from statistics import median
from statistics import stdev
from natsort import natsorted
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation

from script.misc import fileInit
from script.cameraMovementDetection import readData
from script.detectingTimingToObtainH import detectSurgicalField
from script.homographyTransformation import movs2imgs
from script.homographyTransformation import camera2world
from script.homographyTransformation import getHomography
from script.homographyTransformation import convertVideo
from script.cameraMovementDetection import detectMovement
from script.cameraMovementDetection import smoothing
from script.cameraMovementDetection import plotFig
from script.cameraMovementDetection import xmeans

PATH = os.getcwd()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='one path execution')
    parser.add_argument("src_dir", type=str, help="Directory with src data")
    args = parser.parse_args()
    
    SKIP_NUM = 30
    TIME = 300
    TH = 0.5
    MAX = 2

    fileInit.main(args.src_dir)
    print("detecting timing to obtain H-mat.")
    detectSurgicalField.main(args.src_dir)
    time_list_0 = readData.read_data2(os.path.join(args.src_dir, 'txt/detectSurgicalField_time_0.txt'))
    time_list_1 = readData.read_data2(os.path.join(args.src_dir, 'txt/detectSurgicalField_time_1.txt'))
    time_list_2 = readData.read_data2(os.path.join(args.src_dir, 'txt/detectSurgicalField_time_2.txt'))
    FPS = cv2.VideoCapture(glob.glob(os.path.join(args.src_dir, 'videos/input/*'))[0]).get(cv2.CAP_PROP_FPS)
    
    camera_move_list = [0, 100000000]
    with open(os.path.join(args.src_dir, 'txt/cameraMove.txt'), 'w') as f:
        for cm in camera_move_list:
            print(cm, file=f)
    count = 0
    flag = 0
    c1_min = 0
    frame_count = cv2.VideoCapture(glob.glob(os.path.join(args.src_dir, 'videos/input/*'))[0]).get(cv2.CAP_PROP_FRAME_COUNT)
    c1_max = frame_count
    prev_frame = -1
    while True:
        if flag == 0:
            time_list_idx = 0
            time_list = readData.concat_time_list(time_list_0, time_list_1, time_list_2, c1_min, c1_max, FPS)
            if len(time_list) == 0:
                print("error: cannot get Homography in this part.")
                break
        frame = time_list[time_list_idx]
        if prev_frame == frame[0]:
            print("Done.")
            break
        if frame[1] - frame[0] > 60 * FPS:
            frame[1] = int(frame[0] + 60 * FPS)
        start = int(frame[0])
        stop = int(frame[1])
        step = int((frame[1] - frame[0]) / 150)
        movs2imgs.main(args.src_dir, "frame", start, stop, step)
        with open('./script/homographyTransformation/ini.txt', 'w') as f:
            print("PROJ_DIR=\"" + os.path.join(PATH, "script/homographyTransformation/Superglue-COLMAP/") + "\"", file=f)
            print("SUPERGLUE_DIR=\"" + os.path.join(PATH, "script/homographyTransformation/SuperGluePretrainedNetwork") + "\"", file=f)
            print("DATA_DIR=\"../../../" + os.path.join(args.src_dir, "assets/test") + "\"", file=f)
            print("DATASET_PATH=\"" + os.path.join(args.src_dir, "assets/test") + "\"", file=f)

        command = "bash " + os.path.join(PATH, "script/homographyTransformation/Superglue-COLMAP/pipeline.sh")
        p = subprocess.Popen(command, shell=True)
        p.wait()
        with open('./script/homographyTransformation/ini.txt', 'w') as f:
            print("PROJ_DIR=\"" + os.path.join(PATH, "script/homographyTransformation/Superglue-COLMAP/") + "\"", file=f)
            print("SUPERGLUE_DIR=\"" + os.path.join(PATH, "script/homographyTransformation/SuperGluePretrainedNetwork") + "\"", file=f)
            print("DATA_DIR=\"../../../" + os.path.join(args.src_dir, "assets/test") + "\"", file=f)
            print("DATASET_PATH=\"" + os.path.join(args.src_dir, "assets/test") + "\"", file=f)
        command = "bash " + os.path.join(PATH, "script/homographyTransformation/colmap_commandline.sh")
        p = subprocess.Popen(command, shell=True)
        p.wait()
        print(p.returncode)
        if p.returncode != 0:
            print("colmap failed")
            time_list_idx += 1
            if time_list_idx >= len(time_list):
                print("error: cannot get Homography in this part.")
                break
            flag = 1
            continue
        if (camera2world.main(args.src_dir) > 1):
            print("colmap failed")
            time_list_idx += 1
            if time_list_idx >= len(time_list):
                print("error: cannot get Homography in this part.")
                break
            flag = 1
            continue
        else:
            state_getHomography = getHomography.main(args.src_dir)
            if state_getHomography > 0:
                print("colmap failed")
                time_list_idx += 1
                if time_list_idx >= len(time_list):
                    print("error: cannot get Homography in this part.")
                    break
                flag = 1
                continue
            else:
                print("colmap successed")
        
        print("Executing homography transformation.")
        camera_move_list.sort()
        convertVideo.main(args.src_dir)

        print("Camera movement detection")
        detectMovement.main(args.src_dir)
        fig_size = 600
        split_size = 8
        data = readData.read_data(os.path.join(args.src_dir, 'txt/detectMovement_data.txt'))
        detected = []
        detect_flag = 0
        ite = 0
        for sec in range(0, data.shape[0], int(fig_size / split_size)):
            if sec % int(fig_size / split_size) != 0:
                continue
            else:
                if not(os.path.isdir(os.path.join(args.src_dir, 'txt/split', str(sec)))):
                    os.makedirs(os.path.join(args.src_dir, 'txt/split', str(sec)))
                else:
                    for j in range(0, 1000):
                        if os.path.isdir(os.path.join(args.src_dir, 'txt/split_' + str(j))):
                            continue
                        else:
                            shutil.copytree(os.path.join(args.src_dir, 'txt/split'), os.path.join(args.src_dir, 'txt/split_' + str(j)))
                            break
                    shutil.rmtree(os.path.join(args.src_dir, 'txt/split'))
                    os.makedirs(os.path.join(args.src_dir, 'txt/split', str(sec)))
                save_split = data[max(sec - int(fig_size / 2), 0):min(sec + int(fig_size / 2), data.shape[0]), :]
                np.savetxt(os.path.join(args.src_dir, 'txt/split', str(sec), 'detectMovement_data.txt'), save_split)
                ite += 1
                start_frame = max(sec - int(fig_size / 2), 0) * FPS
                end_frame = min(sec + int(fig_size / 2), data.shape[0]) * FPS
                smoothing.main(args.src_dir, sec, start_frame, end_frame)
                th = xmeans.main(args.src_dir, sec, start_frame, end_frame)
                points = plotFig.main(args.src_dir, sec, start_frame, end_frame, th)
                if len(points) > 0 and detect_flag == 0:
                    detect_flag += 1
                    start_mean = sec * FPS
                    end_mean = start_mean + fig_size * ((split_size - 2) / split_size) * FPS
                    detected_points = []
                if detect_flag > 0:
                    detect_flag += 1
                    for p_id, point in enumerate(points):
                        if (point > start_mean) and (point < end_mean):
                            detected_points.append(point)
                            break
                        elif detect_flag  < 3:
                            detect_flag = 100
                if detect_flag >= split_size:
                    detect_flag = 0
                    start_mean = 0
                    end_mean = 0
                    detected.append(detected_points)
                    detected_points = []

        detected.append(detected_points)
        for box_id, box in enumerate(detected):
            if len(box) < split_size / 2:
                detected[box_id] = []
            elif len(box) > 1:
                detected[box_id] = [median(box)]
        camera_move_predict = list(itertools.chain.from_iterable(detected))
        for i, predict in enumerate(camera_move_predict):
            if (predict < camera_move_list[len(camera_move_list) - 2]) and (i != len(camera_move_predict) - 1):
                continue
            elif (predict < camera_move_list[len(camera_move_list) - 2]) and (i == len(camera_move_predict) - 1):
                print("Done.")
                exit(0)
            else:
                if i <= len(camera_move_predict) - 2:
                    c1_min = camera_move_predict[i]
                    c1_max = camera_move_predict[i + 1]
                elif i <= len(camera_move_predict) - 1:
                    c1_min = camera_move_predict[i]
                    c1_max = frame_count
                else:
                    print("Done.")
                    exit(0)
                camera_move_list.append(predict)
                break
        camera_move_list.sort()
        with open(os.path.join(args.src_dir, 'txt/cameraMove.txt'), 'w') as f:
            for cm in camera_move_list:
                print(cm, file=f)


        time_list_idx += 1
        count += 1

        if time_list_idx >= len(time_list):
            print("Done.")
            break
        else:
            flag = 0
            prev_frame = frame[0]  # prev_frame 前のstart
            continue


    
    """
    camera switching
    """
