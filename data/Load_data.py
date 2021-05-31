import os
import numpy as np
import pandas as pd
from skimage import io, transform
import torch
from torch._C import ErrorReport
import torchvision

def string2num1(s):
    n = {
        "L1" : 0,
        "L2" : 1,
        "L3" : 2,
        "L4" : 3, 
        "L5" : 4,
        "T12-L1" : 5,
        "L1-L2" : 6,
        "L2-L3" : 7,
        "L3-L4" : 8,
        "L4-L5" : 9,
        "L5-S1" : 10
    }
    # if not n.get(s, None):
    #     return 1
    return n.get(s, 0)

def string2num2(s):
    n = {
        "v1":1,
        "v2":2,
        "v3":3,
        "v4":4,
        "v5":5
    }
    # if not n.get(s, None):
    #     return 0
    return n.get(s, 0)

def get_list(PATH) :
    files = os.listdir(PATH)
    ids = list()
    for file in files:
        # if file[-3:] != "jpg":
            # ids.append(file)
        if file[-3:] != "txt":
            # txt_list.append(file)
            ids.append(file[0:-4])
    return ids

def read_txt(PATH) :
    bbox = list()
    label = list()
    x = list()
    y = list()
    with open(PATH, "r", encoding = 'utf-8') as f:
        for j in range(11):
            data = f.readline()
            if not data:
                continue
            # print(data)
            p = data.find(',', 0)
            # x = data[0:p]
            y.append(float(data[0:p]))
            q = data.find(',', p+1)
            x.append(float(data[p+1:q]))
            # y = data[p+1:q]
            p = data.find("identification", 0)
            p = p + len("identification': '")
            q = data.find('\'', p)
            str = data[p:q]
            Num = string2num1(str)
            label.append(Num)
            
    h = (max(x) - min(x)) / 9 * 2
    w = h * 1.5
    for i in range(len(x)):
        bbox.append([x[i] - h / 2, y[i] - w / 2, x[i] + h / 2, y[i] + w / 2])
    
    bbox = np.stack(bbox).astype(np.float32)
    label = np.stack(label).astype(np.int32)
    return bbox, label



if __name__ == "__main__":

    PATH = "Data/train/data/"
    


