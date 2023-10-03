import numpy as np
import cv2
import sys
import glob
import argparse
import os
import shutil

VIDEO = 0

def main(src_dir):
    if os.path.isdir(os.path.join(src_dir, 'videos/output')):
        shutil.rmtree(os.path.join(src_dir, 'videos/output'))
        os.makedirs(os.path.join(src_dir, 'videos/output'))
    else:
        os.makedirs(os.path.join(src_dir, 'videos/output'))
    
    video_paths = sorted(glob.glob(os.path.join(src_dir, 'videos/input/*.mp4')))
    homography_paths = sorted(glob.glob(os.path.join(src_dir, 'homographys/test/*')))
    for i, path in enumerate(homography_paths):
        homography_paths[i] = sorted(glob.glob(os.path.join(path, '*.npy')))
    f = open(os.path.join(src_dir, 'txt/cameraMove.txt'), 'r')
    nums = []
    for line in f.readlines():
        try:
            commchar = '#'
            if len(line) == 0 or line[0] == commchar:
                continue
            num = int(float(line))
        except ValueError as e:
            print(e, file=sys.stderr)
            continue
        nums.append(num)
    f.close()
    nums = np.array(nums)
    nums = nums.astype(np.float64)
    for i, video_path in enumerate(video_paths):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            sys.exit()
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        size = (width,height)
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
        save_path = os.path.join(src_dir, 'videos/output/', str(i) + '.mp4')
        save = cv2.VideoWriter(save_path,fourcc,fps,size)
        n= 0
        c = 0
        while True:
            is_image,frame_img = cap.read()
            if is_image:
                if n >= nums[c]:
                    mat = np.load(homography_paths[c][i])
                    c += 1
                perspective_img = cv2.warpPerspective(frame_img, mat, size)
                perspective_img=cv2.resize(perspective_img, size)
                save.write(perspective_img)
            else:
                break
            n += 1
        cap.release()
        save.release()


    
