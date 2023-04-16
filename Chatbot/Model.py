
import tensorflow as tf
import yaml
from underthesea import word_tokenize
import re
from keras.utils import pad_sequences
from process import tokenize, dict_word, label
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

with open('./Chatbot/texts2.yaml', 'r', encoding='utf-8') as f:
    data = yaml.load(f, Loader=yaml.Loader)

dict_label = label(data)
word = tokenize(data)
word_to_id, id_to_word = dict_word(word)
d = []
for i,dt in enumerate (data['chat content'].keys()):
    a = []
    for j in data['chat content'][dt]['question']:
        j = re.sub(r'[^\w\s]', '', str(j)).lower()
        love = word_tokenize(j)
        a.append([word_to_id[_] for _ in love])
    a = pad_sequences(a, maxlen=15,padding='post',truncating='post')
    d.extend([(c,i) for c in a])

import tensorflow as tf
dataset = tf.data.Dataset.from_generator(lambda: iter(d), 
                                         output_types=(tf.int32, tf.int32), 
                                         output_shapes=((15,), ()))

# dataset = dataset.shuffle(buffer_size=50,seed=42)
from keras.losses import SparseCategoricalCrossentropy
from keras.optimizers import  Adam
from keras.models import Model
from keras.layers import Dense, LSTM, Embedding, Input, LeakyReLU, Dropout, Flatten
# from keras.initializers.initializers_v2 import GlorotUniform
dataset = dataset.batch(16)
inputs = Input(shape=(15,))
x = Embedding(input_dim=106, output_dim=100, input_length=15, mask_zero=True)(inputs)
x = LSTM(units=256, activation=LeakyReLU(),dropout=0.2, return_sequences=True)(x)
x = LSTM(units=128, activation=LeakyReLU(), return_sequences=True)(x)
x = Flatten()(x)
x = Dense(units=128, activation=LeakyReLU())(x)
outputs = Dense(units=25, activation='softmax')(x)
model = Model(inputs, outputs)
model.summary()
model.compile(optimizer=Adam(), loss=SparseCategoricalCrossentropy(from_logits=False), metrics=['accuracy'])

model.fit(dataset,epochs=50,verbose=1)

model.save("thang.h5")
