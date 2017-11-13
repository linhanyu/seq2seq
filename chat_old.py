#!/usr/bin/python3
import jieba
import numpy as np

from seq2seq import Seq2Seq, AttentionSeq2Seq
from keras.models import load_model
from gensim.models import Word2Vec

params = {
    'input_shape': (10, 250),
    'output_dim': 250,
    'output_length': 10,
    'hidden_dim': 512,
    'depth': 2
}

s2s_model = Seq2Seq(**params)
s2s_model.load_weights('model/s2s.800.bin')
w2v_model = Word2Vec.load('w2v_model/med250.model.bin')

end_vec = w2v_model.wv['end']

jieba.set_dictionary('../jieba_dict/dict.txt.big')
stopwordset = set()
with open('../jieba_dict/stop_words.txt','r',encoding='utf-8') as sw:
    for line in sw:
        stopwordset.add(line.strip('\n'))

while True:
    try:
        print('input something ...')
        test_str = input()

        test_str = jieba.cut(line, cut_all = False)
        test_str = [word for word in test_str if word not in stopwordset]
        test_str = [w2v_model.wv[word] for word in test_str if word in w2v_model.wv.vocab]
        test_str.append(end_vec)

        while(len(test_str) < 10):
            test_str.append([0] * 250)

        test_str = np.array([test_str])

        res_vec = s2s_model.predict(test_str)[0]
        res_word = [w2v_model.most_similar(positive = [vec], topn = 1) for vec in res_vec]
        print(res_vec)
        print([word[0][0] for word in res_word])
    except Exception as e:
        print(repr(e))
