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

from keras.models import load_model
model = load_model("thang.h5")

import numpy as np
while True:
    input_ = input("ban: ") 
    print('')
    if input_ == "bỏ":
        break
    a =  word_tokenize(re.sub(r'[^\w\s]', '',input_).lower())
    c = []
    for word in a:
        if word not in word_to_id.keys():
            word = word_to_id[word] = 0
        else:
            word = word_to_id[word]
        c.append(word)
    c = [c]
    c = pad_sequences(c, maxlen=15,padding='post',truncating='post')
    d = model.predict(c)
    h = np.argmax(d)
    k = dict_label[h]
    if np.max(d) < 0.5:
        print("xin lỗi tôi không hiểu bạn đang nói gì")
    else:
        print(data['chat content'][k]['answer'])