import numpy as np
import sympy as sp
from scipy.spatial.transform import Rotation
import cv2
import os
import argparse
from natsort import natsorted
import glob
import pathlib
import shutil

from script.homographyTransformation import camera2world


def find_plane(xs, ys, zs):
    r = np.c_[xs, ys, zs]
    c = np.mean(r, axis=0)
    r0 = r - c
    u, s, v = np.linalg.svd(r0)
    nv = v[-1, :]
    ds = np.dot(r, nv)
    param = np.r_[nv, -np.mean(ds)]
    print("plane:", param)
    return param

def calculate_matrix(focal, cx, cy, qw, qx, qy, qz, tx, ty, tz):
    kmat = np.array([[focal, 0, cx],
                        [0, focal, cy],
                        [0, 0, 1]])
    rot = Rotation.from_quat([qx, qy, qz, qw])
    rmat = rot.as_matrix()
    tvec = np.array([tx, ty, tz])
    exparam = np.insert(rmat, 3, [tx, ty, tz], axis=1)
    pmat = np.dot(kmat, exparam)
    return pmat


def onplane_point(img_point, na, nb, nc, nd, pmat):
    u, v = img_point
    x = sp.Symbol('x')
    y = sp.Symbol('y')
    z = sp.Symbol('z')
    s = sp.Symbol('s')
    eq1 = s * u - (pmat[0][0] * x + pmat[0][1] * y + pmat[0][2] * z + pmat[0][3])
    eq2 = s * v - (pmat[1][0] * x + pmat[1][1] * y + pmat[1][2] * z + pmat[1][3])
    eq3 = s - (pmat[2][0] * x + pmat[2][1] * y + pmat[2][2] * z + pmat[2][3])
    eq4 = na * x + nb * y + nc * z + nd
    xans = sp.solve([eq1, eq2, eq3, eq4])[x]
    yans = sp.solve([eq1, eq2, eq3, eq4])[y]
    zans = sp.solve([eq1, eq2, eq3, eq4])[z]
    print("on plane:", [xans, yans, zans])
    return xans, yans, zans


def projection_point(pmat, point):
    point_vec = np.append(np.array(point), 1)
    project_vec = np.dot(pmat, point_vec)
    project_vec = project_vec / project_vec[2]
    project_point = np.delete(project_vec, 2)
    return project_point


def image_conversion(src, dst, imgname, outname):
    src_pts = np.array(src, dtype=np.float32)
    dst_pts = np.array(dst, dtype=np.float32)
    homography = cv2.getPerspectiveTransform(src_pts, dst_pts)
    print(homography)
    img = cv2.imread(imgname)
    h, w, _ = img.shape
    perspective_img = cv2.warpPerspective(img, homography, (w, h))
    cv2.imwrite(outname, perspective_img)

    return homography

def main(src_dir):
    xs, ys, zs, rs, gs, bs = np.flip(np.loadtxt(os.path.join(src_dir, "assets/test/000/points3D.txt"), usecols=[1, 2, 3, 4, 5, 6], comments='#', unpack=True), 1)
    xs = xs[::int(xs.shape[0] / 50000) + 1]
    ys = ys[::int(ys.shape[0] / 50000) + 1]
    zs = zs[::int(zs.shape[0] / 50000) + 1]
    na, nb, nc, nd = find_plane(xs, ys, zs)

    widths, heights, focals, cxs, cys = np.flip(np.loadtxt(os.path.join(src_dir, "assets/test/000/cameras.txt"), usecols=[2, 3, 4, 5, 6], comments='#', unpack=True), 1)
    qws, qxs, qys, qzs, txs, tys, tzs = np.flip(np.loadtxt(os.path.join(src_dir, "assets/test/000/images.txt"), usecols=[1, 2, 3, 4, 5, 6, 7], comments='#', unpack=True)[:, ::2], 1)
    camera_num = widths.shape[0]
    pmats = []   
    for i in range(camera_num):
        pmat = calculate_matrix(focals[i], cxs[i], cys[i], qws[i], qxs[i], qys[i], qzs[i], txs[i], tys[i], tzs[i])
        pmats.append(pmat)

    img_points = [[0, 0], 
                [widths[0], 0],
                [0, heights[0]],
                [widths[0], heights[0]]]
    onplane = []
    for i in range(len(img_points)):
        onplane.append(onplane_point(img_points[i], na, nb, nc, nd, pmats[0]))
    onplane = np.array(onplane)
    edge = camera2world.get_edge_len(onplane)
    std_edge = np.std(edge)

    print("Projection points")
    project_points = []
    for i in range(camera_num):
        project_point = []
        for j in range(len(onplane)):
            project_point.append(projection_point(pmats[i], onplane[j]).tolist())
        print("camera", i, ":", project_point)
        project_points.append(project_point)
    
    images = []
    images2 = natsorted(glob.glob(os.path.join(src_dir, 'assets/test/000/*f.jpg')))
    for f in images2:
        images.append(os.path.split(f)[1])

    if std_edge < 5:
        state = 0
        if os.path.isdir(os.path.join(src_dir, 'homographys/test')):
            for j in range(0, 1000):
                if os.path.isdir(os.path.join(src_dir, 'homographys/test', str(j))):
                    continue
                else:
                    os.makedirs(os.path.join(src_dir, 'homographys/test', str(j)))
                    dir_num = j
                    break
        else:
            os.makedirs(os.path.join(src_dir, 'homographys/test/0'))
            dir_num = 0
        for i, img in enumerate(images):
            homography = image_conversion(project_points[i], img_points, os.path.join(src_dir, 'assets/test/000', img), os.path.join(src_dir, 'assets/test/000', "{}_out.jpg".format(i+1)))   
            np.save(os.path.join(src_dir, 'homographys/test', str(dir_num), str(i)), homography)
    else:
        state = 1

    return state


