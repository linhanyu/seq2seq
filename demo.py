#!/usr/bin/python3
import pickle
import numpy as np
from keras.models import load_model, Model
from keras.layers import Input

word_dict_file = 'data/dict/word_dict.pkl'

# param prepare
word_dict = None
with open(word_dict_file, 'rb') as p:
    word_dict = pickle.load(p)

word_dict_len = len(word_dict)


full_model = load_model('model/s2s.70.bin')
encoder_input = full_model.layers[0]
encoder = full_model.layers[2]
decoder_input = full_model.layers[1]
decoder_lstm = full_model.layers[3]
decoder_dense = full_model.layers[4]

encoder_output, state_h, state_c = encoder(encoder_input)
encoder_state = [state_h, state_c]

encoder_model = Model(encoder_input, encoder_state)

decoder_state_input_h = Input(shape = (word_dict_len, ))
decoder_state_input_c = Input(shape = (word_dict_len, ))
decoder_state_input = [decoder_state_input_h, decoder_state_input_c]

decoder_output, state_h, state_c = decoder_lstm(decoder_input, initial_state = decoder_state_input)
decoder_state = [state_h, state_c]
decoder_output = decoder_dense(decoder_output)

decoder_model = Model([decoder_input] + decoder_state_input, [decoder_output] + decoder_state)

def decode_sequence(input_seq):

    state_value = encoder_model.predict(input_seq)

    target_seq = np.zeros((1, 1, word_dict_len))

    #TODO: insert one-hot of start_sym to target_seq

    stop_condition = False
    decode_sentence = ''

    while not stop_condition:
        output_token, h, c = decoder_model.predict([target_seq] + state_value)

        sample_word = None
        # TODO: sample word from output_token, and concat with decode_sentence
        decode_sentence += sample_word

        if sample_word == end_sym or len(decode_sentence) >= 15:
            stop_condition = True

        target_seq = np.zeros((1, 1, word_dict_len))

        #TODO: insert one-hot of start_sym to target_seq

        state_value = [h, c]

    return decode_sentence
