import os
import shutil
import glob

if __name__ == "__main__":
    files = sorted(glob.glob('../../assets/test/[0-9]*'))
    print(files)
    for d in files:
        dpath = d + '/dump_match_pairs'
        shutil.rmtree(dpath)