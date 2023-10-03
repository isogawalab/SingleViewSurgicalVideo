import os
import itertools
import argparse
import glob
import cv2
import numpy as np

from database import COLMAPDatabase


def create_db_from_SuperGlue_matching():
    parser = argparse.ArgumentParser(description="Create .db from SuperGlue matching.")
    parser.add_argument("src_dir", type=str, default="", help="Directory containing all images")
    parser.add_argument("--database_path", default="database.db", help="Path to COLMAP database (.db)")
    parser.add_argument("--frame_digits", type=int, default=3, help="Number of digits of frame directories' names")
    opt = parser.parse_args()

    #
    # Check if intended folder structure is found.
    #
    if not os.path.exists(opt.src_dir):
        print("ERROR: Image directory does not exist.")
        return
    if os.path.exists(opt.database_path):
        print("ERROR: Database path already exists -- will not modify it.")
        return

    #
    # Open a new empty database.
    #
    db = COLMAPDatabase.connect(opt.database_path)
    # For convenience, try creating all the tables upfront.
    db.create_tables()

    #
    # Find all frame directories.
    # >> Currently, I assume that directry is named as three digits (e.g., 006).
    #
    frame_dirs = sorted(glob.glob(os.path.join(opt.src_dir, "[0-9]" * opt.frame_digits)))
    frame_num = len(frame_dirs)

    #
    # Check images in the first directory (first frame).
    # >> We use names of the first frame images as their camera names.
    #
    image_list = []
    types = ("*.jpg", "*.png")
    for ext in types:
        image_list.extend(sorted(glob.glob(os.path.join(frame_dirs[0], ext))))
    if len(image_list) <= 0:
        print("ERROR: Image directory does not contain any images.")
        return

    camera_num = len(image_list)
    print(f"Found {frame_num} frames of {camera_num} cameras.")

    #
    # Set default images and cameras.
    #
    print(f"Adding default image and cameras.")
    img_ids = []
    for camera_idx, img_filename in enumerate(image_list):
        # Load an image.
        img_name = os.path.basename(img_filename)
        img = cv2.imread(img_filename)

        # Add cameras.
        # Camera model id: https://github.com/colmap/colmap/blob/master/src/base/camera_models.h
        model = 0
        w, h = img.shape[1], img.shape[0]
        f = 1.2 * w
        cx, cy = w / 2.0, h / 2.0
        camera_id = db.add_camera(model, w, h, np.array((f, cx, cy)))
        # print("HELLO1!!!")
        # print(model, w, h, np.array((f, cx, cy)))

        # Add images.
        img_id = db.add_image(img_name, camera_id)
        # print("HELLO2!!!")
        # print(img_name, camera_id)
        img_ids.append(img_id)
        print( f">> {img_name}, id: {camera_id}, model: {model}, ({w}x{h}), {np.array((f, cx, cy))}")

    #
    # Add keypoints.
    #
    print("Adding keypoints (SuperPoints).")
    # Note: npz_kpt_idx is used to get the correct npz files among all pairs
    npz_kpt_idx = [camera_num - 1 - idx for idx in range(camera_num)]
    print(npz_kpt_idx)
    npz_kpt_idx = np.cumsum(npz_kpt_idx) - 1
    print(npz_kpt_idx)
    kpt_offsets_per_camera = []
    for camera_idx in range(camera_num):
        kpt_name = "keypoints0" if camera_idx != camera_num - 1 else "keypoints1"
        all_kpts = []
        kpt_offsets = [0]
        # Accumulate keypoints of all frames of the "camera_idx"th camera.
        for frame_dir in frame_dirs:
            npz_list = sorted(glob.glob(os.path.join(frame_dir, "dump_match_pairs", "*.npz")))
            if len(npz_list) <= 0:
                print("ERROR: Matches directory does not contain any npz files.")
                return
            # Add SuperPoints (keypoints).
            npz = np.load(npz_list[npz_kpt_idx[camera_idx]])
            all_kpts.extend(npz[kpt_name] + 0.5)
            kpt_offsets.append(len(all_kpts))

        all_kpts = np.array(all_kpts)
        db.add_keypoints(img_ids[camera_idx], all_kpts)
        # print("HELLO3!!!")
        # print(img_ids[camera_idx], all_kpts)
        kpt_offsets_per_camera.append(kpt_offsets)
        print(f">> Added {all_kpts.shape[0]} kpts to camera {img_ids[camera_idx]}.")

    #
    # Add SuperGlue matches.
    #
    img_id_pairs = list(itertools.combinations(img_ids, 2))
    for pair_idx, img_id_pair in enumerate(img_id_pairs):
        all_matches = []
        for frame_idx, frame_dir in enumerate(frame_dirs):
            npz_list = sorted(glob.glob(os.path.join(frame_dir, "dump_match_pairs", "*.npz")))
            if len(npz_list) <= 0:
                print("ERROR: Matches directory does not contain any npz files.")
                return

            npz = np.load(npz_list[pair_idx])
            kpts_cnt = npz["keypoints0"].shape[0]
            kpt_indices = np.array([idx for idx in range(kpts_cnt)])
            matches = np.stack([kpt_indices, npz["matches"]], axis=1)
            matches = matches[matches[:, 1] >= 0]  # Keep only valid matches.

            # Add offsets to keypoint indices of matches.
            matches[:, 0] += kpt_offsets_per_camera[img_id_pair[0] - 1][frame_idx]
            matches[:, 1] += kpt_offsets_per_camera[img_id_pair[1] - 1][frame_idx]
            all_matches.extend(matches)

        all_matches = np.array(all_matches)
        db.add_matches(img_id_pair[0], img_id_pair[1], all_matches)
        # print("HELLO4!!!")
        # print(img_id_pair[0], img_id_pair[1], all_matches)

        print(f">> Added {all_matches.shape[0]} matches between {img_id_pair[0]} and {img_id_pair[1]}.")

    # Commit the data to the file.
    db.commit()


if __name__ == "__main__":
    create_db_from_SuperGlue_matching()
