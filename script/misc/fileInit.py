import os
import sys
import cv2
import glob
import shutil
import pathlib
import argparse
import subprocess
import numpy as np
import sympy as sp
import pandas as pd
from statistics import stdev
from natsort import natsorted
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation


def main(src_dir):
    command = "rm -rf " + os.path.join(src_dir, "assets")
    p = subprocess.Popen(command, shell=True)
    p.wait()
    command = "rm -rf " + os.path.join(src_dir, "homographys")
    p = subprocess.Popen(command, shell=True)
    p.wait()
    command = "mkdir " + os.path.join(src_dir, "tmp")
    p = subprocess.Popen(command, shell=True)
    p.wait()
    
    command = "cp " + os.path.join(src_dir, "videos/input/*") + " " + os.path.join(src_dir, "tmp")
    p = subprocess.Popen(command, shell=True)
    p.wait()
    command = "rm -rf " + os.path.join(src_dir, "videos")
    p = subprocess.Popen(command, shell=True)
    p.wait()
    command = "mkdir " + os.path.join(src_dir, "videos")
    p = subprocess.Popen(command, shell=True)
    p.wait()
    command = "mkdir " + os.path.join(src_dir, "videos/input")
    p = subprocess.Popen(command, shell=True)
    p.wait()
    command = "mv " + os.path.join(src_dir, "tmp/*") + " " + os.path.join(src_dir, "videos/input")
    p = subprocess.Popen(command, shell=True)
    p.wait()
    command = "rm -rf " + os.path.join(src_dir, "tmp")
    p = subprocess.Popen(command, shell=True)
    p.wait()

    command = "mkdir " + os.path.join(src_dir, "assets") +\
                " & mkdir " + os.path.join(src_dir, "homographys")
    p = subprocess.Popen(command, shell=True)
    p.wait()
    
    command = "mkdir " + os.path.join(src_dir, "fig")
    p = subprocess.Popen(command, shell=True)
    p.wait()
    command = "mkdir " + os.path.join(src_dir, "txt")
    p = subprocess.Popen(command, shell=True)
    p.wait()
