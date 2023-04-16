
import yaml
from underthesea import word_tokenize
import re


with open('./Chatbot/texts2.yaml', 'r', encoding='utf-8') as f:
    data = yaml.load(f, Loader=yaml.Loader)

def label(data):
    dict_lables = {}
    for i, label in enumerate( data['chat content'].keys()):
        dict_lables[i] = label
    return dict_lables

def tokenize(data):
    ques = []
    list_ = []
    for i in  data['chat content'].keys():
        for j in data['chat content'][i]['question']:
            # ques.extend(j)
            j = (word_tokenize(re.sub(r'[^\w\s]', '', str(j)).lower()))
            ques.append(j)
    [list_.extend(q) for q in ques]
    return list(set(list_))

def dict_word(data):
    word_to_id = {}
    id_to_word = {}
    for i in data:
        if i not in word_to_id:
            id = len(word_to_id) + 1
            word_to_id[i] = id
            id_to_word[id] = i
    return word_to_id, id_to_word


