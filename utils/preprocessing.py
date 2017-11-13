#!/vsr/bin/python3
import jieba
from gensim.models import Word2Vec
from keras.utils import np_utils

start_sym = 'start'
end_sym = 'end'
unk_sym = '不明'
pad_sym = 'pad'

w2v_model = Word2Vec.load('w2v_model/med250.model.bin')
start_wv = w2v_model.wv[start_sym].tolist()
end_wv = w2v_model.wv[end_sym].tolist()
unk_wv = w2v_model.wv[unk_sym].tolist()
pad_wv = w2v_model.wv[pad_sym].tolist()

# jieba setup
jieba.set_dictionary('../jieba_dict/dict.txt.big')
stopwordset = set()
with open('../jieba_dict/stop_words.txt','r',encoding='utf-8') as sw:
    for line in sw:
        stopwordset.add(line.strip('\n'))

def trim(line):
    return ''.join(line.split(' '))

def segment(line):
    words = []
    line_split = jieba.cut(line, cut_all = False)
    for word in line_split:
        if word not in stopwordset:
            words.append(word)

    return words

def to_wv_seq(word_seq):
    vec_seq = []
    for word in word_seq:
        if word not in w2v_model.wv.vocab:
            vec_seq.append(unk_wv)
        else:
            vec_seq.append(w2v_model.wv[word].tolist())

    return vec_seq


def to_fixed_wv_seq(word_seq, length):
    vec_seq = []
    for word in word_seq:

        if word not in w2v_model.wv.vocab:
            vec_seq.append(unk_wv)
        else:
            vec_seq.append(w2v_model.wv[word].tolist())

    vec_seq.append(end_wv)

    while len(vec_seq) < length:
        vec_seq.append(pad_wv)

    return vec_seq[:length]


def to_onehot(data, num_classes):
    return np_utils.to_categorical(data, num_classes = num_classes)
