#!/usr/bin/python3
import pickle
import numpy as np

from keras.models import load_model, Model
from keras.layers import Input

from utils.preprocessing import *

# static param
word_dict_file = 'data/dict/word_dict.pkl'
line_length = 15

# param prepare
word_dict = None
with open(word_dict_file, 'rb') as p:
    word_dict = pickle.load(p)

word_dict_len = len(word_dict)

# init model
full_model = load_model('model/s2s.80.bin')
encoder_input = Input(shape = (None, 250))
encoder = full_model.layers[2]
decoder_input = Input(shape = (None, word_dict_len))
decoder_lstm = full_model.layers[3]
decoder_dense = full_model.layers[4]

encoder_output, state_h, state_c = encoder(encoder_input)
encoder_state = [state_h, state_c]

encoder_model = Model(encoder_input, encoder_state)

decoder_state_input_h = Input(shape = (256, ))
decoder_state_input_c = Input(shape = (256, ))
decoder_state_input = [decoder_state_input_h, decoder_state_input_c]

decoder_output, state_h, state_c = decoder_lstm(decoder_input, initial_state = decoder_state_input)
decoder_state = [state_h, state_c]
decoder_output = decoder_dense(decoder_output)

decoder_model = Model([decoder_input] + decoder_state_input, [decoder_output] + decoder_state)

def decode_sequence(input_seq):

    state_value = encoder_model.predict(input_seq)

    target_seq = np.zeros((1, 1, word_dict_len))

    # insert one-hot of start_sym to target_seq
    target_seq[0, 0, word_dict.index(start_sym)] = 1

    stop_condition = False
    decode_sentence = ''

    i = 0
    while not stop_condition:
        output_token, h, c = decoder_model.predict([target_seq] + state_value)

        sample_word = word_dict[np.argmax(output_token[0, -1, :])]
        decode_sentence += sample_word

        if sample_word == end_sym or i >= line_length:
            stop_condition = True

        target_seq = np.zeros((1, 1, word_dict_len))
        target_seq[0, 0, word_dict.index(start_sym)] = 1

        state_value = [h, c]
        i += 1
        # print('gen response: ' + decode_sentence)

    return decode_sentence

if __name__ == '__main__':

    print('start testing')
    test_str = '為什麼 聖結石 會被酸而 這群人 不會？'
    test_str = segment(trim(test_str))
    test_str = to_fixed_wv_seq(test_str, length = line_length)

    response = decode_sequence(np.array([test_str]))
    print('full respones: ' + response)

    print('start testing')
    test_str = '為什麼慶祝228會被罵可是慶端午不會？'
    test_str = segment(trim(test_str))
    test_str = to_fixed_wv_seq(test_str, length = line_length)

    response = decode_sequence(np.array([test_str]))
    print('full respones: ' + response)

    print('start testing')
    test_str = '你怎麼那麼醜？'
    test_str = segment(trim(test_str))
    test_str = to_fixed_wv_seq(test_str, length = line_length)

    response = decode_sequence(np.array([test_str]))
    print('full respones: ' + response)
