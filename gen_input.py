#!/usr/bin/python3
import os
import gc
import pickle
import sys
import numpy as np
from copy import deepcopy
from utils.preprocessing import *

def log(tag, msg):
    print(tag.ljust(10) + ': ' + msg)

raw_data = sys.argv[1]
dst_file_x = 'data/input/train.x'
dst_file_y = 'data/input/train.y'
dst_file_din = 'data/input/train.din'
word_dict_file = 'data/dict/word_dict.pkl'
max_lines = 500
line_length = 15

word_dict = set()
word_dict.add(start_sym)
word_dict.add(end_sym)
word_dict.add(unk_sym)
word_dict.add(pad_sym)

seg_data = []

log('START', 'load text')
src_file = open(raw_data, 'r')
while True:
    line = src_file.readline()
    if not line:
        break

    qa = [segment(trim(single_str.rstrip())) for single_str in line.split('\t')]
    assert len(qa) == 2, 'unexpected length: ' + len(qa)
    seg_data.append(qa)

src_file.close()
log('END', 'load text')

# dump train_x, each 1000 line dump to a single file
log('START', 'wv processing')
for page in range(0, len(seg_data), max_lines):
    train_x = [to_fixed_wv_seq(qa[0], length = line_length) for qa in seg_data[page:page + max_lines]]

    assert np.array(train_x).shape == (max_lines, line_length, 250), 'unexpected shape: ' + str(np.array(train_x).shape)
    np.save(dst_file_x + '.' + str(int(page / max_lines)), np.array(train_x))
    log('PROGRESS', 'process ' + str(page + max_lines) + ' lines')
log('END', 'wv processing')

# free memory
del stopwordset
gc.collect()

# gen dict
log('START', 'generate word dict')
for qa in seg_data:
    for word in qa[1]:
        if word in w2v_model.wv.vocab:
            word_dict.add(word)
log('END', 'generate word dict')
log('INFO', 'length of word dict: ' + str(len(word_dict)))

word_dict = list(word_dict)

# dump word dict
log('START', 'dump word dict')
with open(word_dict_file, 'wb') as p:
    pickle.dump(word_dict, p)
log('END', 'dump word dict, path: ' + word_dict_file)

del w2v_model
gc.collect()

# idx, val -> val: idx
word_dict = {el: idx for idx, el in enumerate(word_dict)}
word_dict_len = len(word_dict)

cmd = 'echo ' + str(word_dict_len) + ' > word_dict_len'
log('SHELL', cmd)
os.system(cmd)

# dump train_y, each 1000 line dump to a single file
log('START', 'one-hot processing')
for page in range(0, len(seg_data), max_lines):
    train_y = [ \
        [word_dict[word] if word in word_dict else word_dict[unk_sym] for word in qa[1]] \
        for qa in seg_data[page:page + max_lines] \
    ]

    # insert end symbol and fixd length
    for i in range(len(train_y)):
        train_y[i] += [word_dict[end_sym]]

        while len(train_y[i]) < line_length:
            train_y[i] += [word_dict[pad_sym]]

        train_y[i] = train_y[i][:line_length]

    # handle decoder input
    train_d_in = deepcopy(train_y)
    for i in range(len(train_d_in)):
        train_d_in[i] = [word_dict[start_sym]] + train_d_in[i]
        train_d_in[i] = train_d_in[i][:line_length]

    # one-hot encoding
    for i in range(len(train_y)):
        train_y[i] = to_onehot(train_y[i], num_classes = word_dict_len)
        train_d_in[i] = to_onehot(train_d_in[i], num_classes = word_dict_len)

    assert np.array(train_y).shape == (max_lines, line_length, word_dict_len), 'unexpected shape: ' + str(np.array(train_y).shape)
    assert np.array(train_d_in).shape == (max_lines, line_length, word_dict_len), 'unexpected shape: ' + str(np.array(train_d_in).shape)
    np.save(dst_file_y + '.' + str(int(page / max_lines)), np.array(train_y))
    np.save(dst_file_din + '.' + str(int(page / max_lines)), np.array(train_d_in))
    log('PROGRESS', 'process ' + str(page + max_lines) + ' lines')


log('END', 'one-hot processing')
