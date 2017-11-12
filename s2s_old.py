import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

import pickle
import numpy as np
from seq2seq.models import Seq2Seq, AttentionSeq2Seq

from keras.optimizers import RMSprop, Adam
from keras.callbacks import Callback

class SaveBestModel(Callback):
    def on_train_begin(self, logs = {}):
        self.min = 100

    def on_epoch_end(self, epoch, logs = None):
        if logs.get('val_loss') < self.min:
            self.min = logs.get('val_loss')
            self.model.save_weights('model/s2s.best.bin')
            print('\nsave best model, val_loss = ' + str(self.min))

        if epoch % 100 == 0:
            self.model.save('model/s2s.' + str(epoch) + '.bin')


params = {
    'input_shape': (10, 250),
    'output_dim': 250,
    'output_length': 10,
    'hidden_dim': 512,
    'depth': 2
}

x_train = None
y_train = None

print('load training data')
with open('data/w2vdata/data.11.conv.x', 'rb') as f:
    x_train = pickle.load(f)

with open('data/w2vdata/data.11.conv.y', 'rb') as f:
    y_train = pickle.load(f)

print(x_train.shape)
print(y_train.shape)

cb = SaveBestModel()
model = Seq2Seq(**params)
model.compile(loss = 'mse', optimizer = Adam(clipvalue = 3))

model.summary()

print('start to training')
model.fit(x_train, y_train, epochs = 1000, batch_size = 128, validation_split = 0.2, callbacks = [cb])

model.save_weights('model/s2s.bin')
