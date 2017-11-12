#!/usr/bin/python3
import numpy as np
import random
import pickle

data_path_prefix = 'data/input/train.x'
label_path_prefix = 'data/input/train.y'
din_path_prefix = 'data/input/train.din'

def get_batch(batch_size = 128):
    file_idx = random.randint(0, 19)
    select = random.sample(range(1000), batch_size)

    full_data = None
    full_label = None
    full_din = None
    with open(data_path_prefix + '.' + str(file_idx), 'rb') as p:
        full_data = pickle.load(p)

    with open(label_path_prefix + '.' + str(file_idx), 'rb') as p:
        full_label = pickle.load(p)

    with open(din_path_prefix + '.' + str(file_idx), 'rb') as p:
        full_din = pickle.load(p)

    data = []
    label = []
    din = []

    for idx in select:
        data.append(full_data[idx])
        label.append(full_label[idx])
        din.append(full_din[idx])

    return np.array(data), np.array(din), np.array(label)
