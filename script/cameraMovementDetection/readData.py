import numpy as np
import sys
import pandas as pd

FPS = 30

def read_data(path):
    f = open(path, 'r')
    data = []
    x = 0
    for line in f.readlines():
        try:
            commchar = '#'
            if len(line) == 0 or line[0] == commchar:
                continue
            num = float(line)
        except ValueError as e:
            print(e, file=sys.stderr)
            continue
        data.append([x, num])
        x += FPS
    f.close()
    data = np.array(data)
    return data

def read_data2(path):
    f = open(path, 'r')
    y = []
    for line in f.readlines():
        try:
            commchar = '#'
            if len(line) == 0 or line[0] == commchar:
                continue
            num = [float(s) for s in line.split()]
        except ValueError as e:
            print(e, file=sys.stderr)
            continue
        y.append(num)
    f.close()
    y = np.array(y)
    return y

def concat_time_list(time_list_0, time_list_1, time_list_2, min, max, fps):
    time_list = []
    for l in (time_list_0, time_list_1, time_list_2):
        tl = []
        for t in l:
            s = 0
            e = 0
            if t[0] <= min and t[1] >= max:
                s = min
                e = max
            elif t[0] <= min and t[1] <= min:
                continue
            elif t[0] <= min and t[1] >= min:
                s = min
                e = t[1]
            elif t[0] >= max and t[1] >= max:
                continue
            elif t[0] <= max and t[1] >= max:
                s = t[0]
                e = max
            elif t[0] >= min and t[0] <= max and t[1] >= min and t[1] < max:
                s = t[0]
                e = t[1]
            
            if e - s > 10 * fps:
                tl.append([s, e])
        sorted(tl)
        time_list.extend(tl)

    return time_list

def concat_list(timebox):
    is_elem = 0
    for box_id in range(len(timebox) - 1, -1, -1):
        if len(timebox[box_id]) == 0 and is_elem == 0:
            continue
        elif len(timebox[box_id]) == 0 and is_elem == 1:
            is_elem = 0
        elif len(timebox[box_id]) != 0 and is_elem == 0:
            is_elem = 1
        elif len(timebox[box_id]) != 0 and is_elem == 1:
            elem = timebox.pop(box_id + 1)
            timebox[box_id].extend(elem)
            print(f"concat:{timebox}")
        else:
            continue
    return timebox

