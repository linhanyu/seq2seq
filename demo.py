#!/usr/bin/python3
import numpy as np
from keras.models import load_model, Model
from keras.layers import Input

full_model = load_model('model/s2s.100.bin')
encoder_input = full_model.layers[0]
encoder = full_model.layers[2]
decoder_input = full_model.layers[1]
decoder_lstm = full_model.layers[3]
decoder_dense = full_model.layers[4]

encoder_output, state_h, state_c = encoder(encoder_input)
encoder_state = [state_h, state_c]

encoder_model = Model(encoder_input, encoder_state)

decoder_state_input_h = Input(shape = (3561, ))
decoder_state_input_c = Input(shape = (3561, ))
decoder_state_input = [decoder_state_input_h, decoder_state_input_c]

decoder_output, state_h, state_c = decoder_lstm(decoder_input, initial_state = decoder_state_input)
decoder_state = [state_h, state_c]
decoder_output = decoder_dense(decoder_output)

decoder_model = Model([decoder_input] + decoder_state_input, [decoder_output] + decoder_state)

def decode_sequence(input_seq):

    state_value = encoder_model.predict(input_seq)

    target_seq = np.zeros((1, 1, 3561))

    #TODO: insert one-hot of start_sym to target_seq#


