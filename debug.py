# -*- coding: utf-8 -*-
import re
import os
import sys
import model
import torch
import codecs
import numpy as np
import itertools
from torch.autograd import Variable
from loader import *
import json


def evaluate(model, datas):
    data = datas[0]
    # print(type(data))

    # ground_truth_id = data['tags']
    words = data['str_words']
    chars2 = data['chars']
    caps = data['caps']

    chars2_sorted = sorted(chars2, key=lambda p: len(p), reverse=True)
    d = {}
    for i, ci in enumerate(chars2):
        for j, cj in enumerate(chars2_sorted):
            if ci == cj and not j in d and not i in list(d.values()):
                d[j] = i
                continue

    chars2_length = [len(c) for c in chars2_sorted]
    char_maxl = max(chars2_length)
    chars2_mask = np.zeros((len(chars2_sorted), char_maxl), dtype='int')
    for i, c in enumerate(chars2_sorted):
        chars2_mask[i, :chars2_length[i]] = c
    chars2_mask = Variable(torch.LongTensor(chars2_mask))

    dwords = Variable(torch.LongTensor(data['words']))
    dcaps = Variable(torch.LongTensor(caps))

    val, out = model(dwords, chars2_mask, dcaps, chars2_length, d)
    return out

train_sentences = load_sentences("dataset/aminer_segment/aminer_train.dat", lower=True, zeros=0)
update_tag_scheme(train_sentences, 'iob')
dico_words_train = word_mapping(train_sentences, lower=True)[0]
dico_words, word_to_id, id_to_word = augment_with_pretrained(
    dico_words_train.copy(),
    "models/glove.6B.100d.txt", None
)

dico_chars, char_to_id, id_to_char = char_mapping(train_sentences)
dico_tags, tag_to_id, id_to_tag = tag_mapping(train_sentences)

model = torch.load('models/aminer', map_location='cpu')
model.use_gpu = 0


def test(txt):
    txt = txt.lower()
    txt = txt.rstrip()
    word = txt.split()
    print(word)
    file = open('dataset/api.txt', 'w')
    for w in word:
        file.write(w + ' O\n')
    file.write('\n-DOCSTART- -X- O O')
    file.close()

    # test_file = open("dataset/train_test.dat", "r", encoding="utf8")
    # data = []
    # for line in test_file.readlines():
    #     # print(line)
    #     word = line.split()
    #     if len(word) < 2:
    #         break
    #     temp_word = [word[0], word[1]]
    #     data.append(temp_word)

    sentences = load_sentences('dataset/api.txt', 1, 1)
    # print(sentences)
    # update_tag_scheme(sentences, "iob")
    # [3, 4, 0, 6, 7, 9, 13, 10, 0, 0, 0, 0, 16]

    input_data = prepare_dataset(
        sentences, word_to_id, char_to_id, tag_to_id, lower=True
    )
    prediction_id = evaluate(model=model, datas=input_data)
    print(prediction_id)
    print(id_to_tag)

    prediction_tag = []
    for id in prediction_id:
        prediction_tag.append(id_to_tag[id])
    print(sentences)
    # print(input_data)

    # per = []
    # con = []
    # date = []
    # org = []
    # key = []
    # o = []
    # # dicts = [loc, per, con, date, org, key, o]
    # for w, id in zip(word, prediction_id):
    #     if id == tag_to_id['I-ORG'] or id == tag_to_id['B-ORG']:
    #         org.append(w)
    #     elif id == tag_to_id['I-KEY'] or id == tag_to_id['B-KEY']:
    #         key.append(w)
    #     elif id == tag_to_id['I-PER'] or id == tag_to_id['B-PER']:
    #         per.append(w)
    #     elif id == tag_to_id['I-CON'] or id == tag_to_id['B-CON']:
    #         con.append(w)
    #     elif id == tag_to_id['B-DATE']:
    #         date.append(w)
    #     else:
    #         o.append(w)
    #
    # ans = {
    #     'PER': per,
    #     'CON': con,
    #     'DATE': date,
    #     'ORG': org,
    #     'KEY': key,
    #     'O': o
    # }
    # print(ans)
    # return ans

    label = ['DATE', 'ORG', 'KEY', 'PER', 'CON']
    ans = {
        'PER': [],
        'CON': [],
        'DATE': [],
        'ORG': [],
        'KEY': [],
        'O': []
    }

    stat = None
    tmp = ''
    for w, i in zip(word, prediction_id):
        found_b = False
        for s in label:
            if i == tag_to_id['B-' + s]:
                if stat is not None:
                    ans[stat].append(tmp)
                stat = s
                tmp = w
                found_b = True
                break
        if found_b:
            continue
        else:
            if stat is not None and 'I-' + stat in tag_to_id and i == tag_to_id['I-' + stat]:
                tmp += ' ' + w
            else:
                if stat is not None:
                    ans[stat].append(tmp)
                stat = 'O'
                tmp = w
    if stat is not None:
        ans[stat].append(tmp)
    return ans

if __name__ == "__main__":
    print('start')
    txt = "Trend Micro paper published by Lantz Marilyn S that are made in 1902"
    test(txt)
