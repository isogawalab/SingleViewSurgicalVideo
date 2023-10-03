import cv2
import os
import argparse
from natsort import natsorted
import shutil

def save_frame_range(video_path, start_frame, stop_frame, step_frame,
                     dir_path, basename, ext='jpg'):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return

    os.makedirs(dir_path, exist_ok=True)

    digit = len(str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))

    idx = 0
    for n in range(start_frame, stop_frame, step_frame):
        cap.set(cv2.CAP_PROP_POS_FRAMES, n)
        ret, frame = cap.read()
        if ret:
            frame_dir = str(idx).zfill(3)
            os.makedirs(os.path.join(dir_path, frame_dir), exist_ok=True)
            base_path = os.path.join(dir_path, frame_dir, basename)
            cv2.imwrite('{}_{}f.{}'.format(base_path, str(n).zfill(digit), ext), frame)
            idx += 1
        else:
            return


def save_sec_range(video_path, start_sec, stop_sec, step_sec,
                         dir_path, basename, ext='jpg'):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return

    os.makedirs(dir_path, exist_ok=True)

    digit = len(str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps_inv = 1 / fps

    idx = 0
    sec = start_sec
    while sec < stop_sec:
        n = round(fps * sec)
        cap.set(cv2.CAP_PROP_POS_FRAMES, n)
        ret, frame = cap.read()
        if ret:
            frame_dir = str(idx).zfill(3)
            os.makedirs(os.path.join(dir_path, frame_dir), exist_ok=True)
            base_path = os.path.join(dir_path, frame_dir, basename)
            cv2.imwrite('{}_{:.2f}s.{}'.format(base_path, n * fps_inv, ext), frame)
            idx += 1
        else:
            return
        sec += step_sec

def main(src_dir, mode, start, stop, step):
    if os.path.isdir(os.path.join(src_dir, 'assets/test')):
        for j in range(0, 1000):
            if os.path.isdir(os.path.join(src_dir, 'assets/', str(j))):
                continue
            else:
                shutil.copytree(os.path.join(src_dir, 'assets/test'), os.path.join(src_dir, 'assets/', str(j)))
                break
        shutil.rmtree(os.path.join(src_dir, 'assets/test'))
        os.makedirs(os.path.join(src_dir, 'assets/test'))
    else:
        os.makedirs(os.path.join(src_dir, 'assets/test'))
    videos = natsorted(os.listdir(os.path.join(src_dir, 'videos/input')))
    if mode == "frame":
        for i, video in enumerate(videos):
            save_frame_range(os.path.join(src_dir, 'videos/input', video),
                            start, stop, step,
                            os.path.join(src_dir, 'assets/test'), str(i + 1))
            print('Video {} is done'.format(i + 1))
    elif mode == "sec":
        for i, video in enumerate(videos):
            save_sec_range(os.path.join(src_dir, 'videos/input', video),
                                start, stop, step,
                                os.path.join(src_dir, 'assets/test'), str(i + 1))
            print('Video {} is done'.format(i + 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert videos to images by specifying frames or seconds.')
    parser.add_argument("mode", type=str, help="Whether to specify the range in frames or seconds")
    parser.add_argument("video_dir", type=str, help="Directory with videos")
    parser.add_argument("start", type=int, help="Start point")
    parser.add_argument("stop", type=int, help="End point")
    parser.add_argument("step", type=int, help="Step")
    parser.add_argument("dst_dir", type=str, help="Output directory")
    args = parser.parse_args() 

    videos = natsorted(os.listdir(args.video_dir))
    print("videos:", videos)

    if args.mode == "frame":
        for i, video in enumerate(videos):
            save_frame_range(os.path.join(args.video_dir, video),
                            args.start, args.stop, args.step,
                            args.dst_dir, str(i + 1))
            print('Video {} is done'.format(i + 1))
    elif args.mode == "sec":
        for i, video in enumerate(videos):
            save_sec_range(os.path.join(args.video_dir, video),
                                args.start, args.stop, args.step,
                                args.dst_dir, str(i + 1))
            print('Video {} is done'.format(i + 1))
