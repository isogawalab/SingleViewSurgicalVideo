import os
import glob
import argparse
import itertools

def get_img_pairs(img_dir):
    types = ("*.jpg", "*.png")
    filenames = []
    for ext in types:
        filenames.extend(sorted(glob.glob(os.path.join(img_dir, ext))))
    filenames = [os.path.basename(filename) for filename in filenames]        
    pairs = list(itertools.combinations(filenames, 2))
    return pairs, filenames


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate image pair list (.txt)")
    parser.add_argument("img_dir", type=str, default="", help="Path to a directory containing images")
    parser.add_argument("dst_txt", default="pairs.txt", help="Image pairs in .txt (default: pairs.txt)")
    opt = parser.parse_args()

    pairs, filenames = get_img_pairs(opt.img_dir)

    print(f">> Found {len(filenames)} images and {len(pairs)} pairs in {opt.img_dir}.")
    print(f">> Writing {len(pairs)} pairs to {opt.dst_txt}")
    with open(opt.dst_txt, "w") as f:
        for pair in pairs:
            f.write(f"{pair[0]} {pair[1]}\n")
    print(f">> Done with writing {opt.dst_txt}.")

