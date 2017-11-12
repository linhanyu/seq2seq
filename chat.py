#!/usr/bin/python3
import numpy as np
import pickle
import jieba
from keras.models import load_model
from gensim.models import Word2Vec

jieba.set_dictionary('../jieba_dict/dict.txt.big')
stopwordset = set()
with open('../jieba_dict/stop_words.txt','r',encoding='utf-8') as sw:
    for line in sw:
        stopwordset.add(line.strip('\n'))

word_dict = None
with open('data/dict/word_dict', 'rb') as f:
    word_dict = pickle.load(f)

w2v_model = Word2Vec.load('w2v_model/med250.model.bin')
s2s_model = load_model('model/s2s.100.bin')

end_vec = w2v_model.wv['end']
unk_vec = w2v_model.wv['不明']

while True:
    try:
        print('input something ...')
        test_str = input()

        test_str = jieba.cut(line, cut_all = False)
        test_str = [word for word in test_str if word not in stopwordset]
        test_str = [w2v_model.wv[word] for word in test_str if word in w2v_model.wv.vocab]
        test_str.append(end_vec)

        while(len(test_str) < 10):
            test_str.append(end_vec)

        test_str = np.array([test_str])

        res_vec = s2s_model.predict(test_str)[0].tolist()
        res_word = [vec.index(max(vec)) for vec in res_vec]
        print(res_word)
    except Exception as e:
        print(repr(e))
