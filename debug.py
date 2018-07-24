# -*- coding: utf-8 -*-
import re
import os
import sys
import model
import torch
import codecs
import numpy as np
from torch.autograd import Variable


def zero_digits(s):
    """
    Replace every digit in a string by a zero.
    """
    return re.sub('\d', '0', s)


def cap_feature(s):
    """
    Capitalization feature:
    0 = low caps
    1 = all caps
    2 = first letter caps
    3 = one capital (not first letter)
    """
    if s.lower() == s:
        return 0
    elif s.upper() == s:
        return 1
    elif s[0].upper() == s[0]:
        return 2
    else:
        return 3


def create_dico(item_list):
    """
    Create a dictionary of items from a list of list of items.
    """
    assert type(item_list) is list
    dico = {}
    for items in item_list:
        for item in items:
            if item not in dico:
                dico[item] = 1
            else:
                dico[item] += 1
    return dico


def create_mapping(dico):
    """
    Create a mapping (item to ID / ID to item) from a dictionary.
    Items are ordered by decreasing frequency.
    """
    sorted_items = sorted(list(dico.items()), key=lambda x: (-x[1], x[0]))
    id_to_item = {i: v[0] for i, v in enumerate(sorted_items)}
    item_to_id = {v: k for k, v in list(id_to_item.items())}
    return item_to_id, id_to_item


def word_mapping(sentences, lower):
    """
    Create a dictionary and a mapping of words, sorted by frequency.
    """
    words = [[x[0].lower() if lower else x[0] for x in s] for s in sentences]
    dico = create_dico(words)

    dico['<PAD>'] = 10000001
    dico['<UNK>'] = 10000000
    dico = {k:v for k,v in list(dico.items()) if v>=3}
    word_to_id, id_to_word = create_mapping(dico)

    print("Found %i unique words (%i in total)" % (
        len(dico), sum(len(x) for x in words)
    ))
    return dico, word_to_id, id_to_word


def char_mapping(sentences):
    """
    Create a dictionary and mapping of characters, sorted by frequency.
    """
    chars = ["".join([w[0] for w in s]) for s in sentences]
    dico = create_dico(chars)
    dico['<PAD>'] = 10000000
    # dico[';'] = 0
    char_to_id, id_to_char = create_mapping(dico)
    print("Found %i unique characters" % len(dico))
    return dico, char_to_id, id_to_char


def tag_mapping(sentences):
    """
    Create a dictionary and a mapping of tags, sorted by frequency.
    """
    tags = [[word[-1] for word in s] for s in sentences]
    dico = create_dico(tags)
    dico[model.START_TAG] = -1
    dico[model.STOP_TAG] = -2
    tag_to_id, id_to_tag = create_mapping(dico)
    print("Found %i unique named entity tags" % len(dico))
    return dico, tag_to_id, id_to_tag


def load_sentences(path, lower, zeros):
    """
    Load sentences. A line must contain at least a word and its tag.
    Sentences are separated by empty lines.
    """
    sentences = []
    sentence = []
    for line in codecs.open(path, 'r', 'utf-8'):
        line = zero_digits(line.rstrip()) if zeros else line.rstrip()
        if not line:
            if len(sentence) > 0:
                if 'DOCSTART' not in sentence[0][0]:
                    sentences.append(sentence)
                sentence = []
        else:
            word = line.split()
            assert len(word) >= 2
            sentence.append(word)
    if len(sentence) > 0:
        if 'DOCSTART' not in sentence[0][0]:
            sentences.append(sentence)
    return sentences


def prepare_data(sentence, word_to_id, char_to_id, tag_to_id, lower=True):
    """
    Prepare the dataset. Return a list of lists of dictionaries containing:
        - word indexes
        - word char indexes
        - tag indexes
    """
    def f(x): return x.lower() if lower else x
    # data = []
    str_words = [w[0] for w in sentence]
    words = [word_to_id[f(w) if f(w) in word_to_id else '<UNK>']
        for w in str_words]
    # Skip characters that are not in the training set
    chars = [[char_to_id[c] for c in w if c in char_to_id]
        for w in str_words]
    caps = [cap_feature(w) for w in str_words]
    # tags = [tag_to_id[w[-1]] for w in sentence]
    tags = ['' for w in sentence]
    data = {
        'str_words': str_words,
        'words': words,
        'chars': chars,
        'caps': caps,
        'tags': tags,
    }
    return data


def augment_with_pretrained(dictionary, ext_emb_path, words):
    """
    Augment the dictionary with words that have a pretrained embedding.
    If `words` is None, we add every word that has a pretrained embedding
    to the dictionary, otherwise, we only add the words that are given by
    `words` (typically the words in the development and test sets.)
    """
    print('Loading pretrained embeddings from %s...' % ext_emb_path)
    assert os.path.isfile(ext_emb_path)

    # Load pretrained embeddings from file
    pretrained = set([
        line.rstrip().split()[0].strip()
        for line in codecs.open(ext_emb_path, 'r', 'utf-8')
        if len(ext_emb_path) > 0
    ])

    # We either add every word in the pretrained file,
    # or only words given in the `words` list to which
    # we can assign a pretrained embedding
    if words is None:
        for word in pretrained:
            if word not in dictionary:
                dictionary[word] = 0
    else:
        for word in words:
            if any(x in pretrained for x in [
                word,
                word.lower(),
                re.sub('\d', '0', word.lower())
            ]) and word not in dictionary:
                dictionary[word] = 0

    word_to_id, id_to_word = create_mapping(dictionary)
    return dictionary, word_to_id, id_to_word


def evaluating(model, data):

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

if __name__ == "__main__":
    train_sentences = load_sentences("dataset/train.dat", lower=True, zeros=0)
    dico_words_train = word_mapping(train_sentences, lower=True)[0]
    dico_words, word_to_id, id_to_word = augment_with_pretrained(
        dico_words_train.copy(),
        "models/glove.6B.100d.txt", None
    )

    txt = "Tsinghua 2012 Jie Tang"
    txt = txt.rstrip()
    word = txt.split()
    data = []
    for w in word:
        temp_word = []
        temp_word.append(w)
        temp_word.append('')
        data.append(temp_word)

    dico_chars, char_to_id, id_to_char = char_mapping(train_sentences)
    dico_tags, tag_to_id, id_to_tag = tag_mapping(train_sentences)

    input_data = prepare_data(
        data, word_to_id, char_to_id, tag_to_id, lower=True
    )

    model = torch.load('models/test', map_location='cpu')
    model.use_gpu = 0
    prediction_id = evaluating(model=model, data=input_data)
    prediction_tag = []
    for id in prediction_id:
        prediction_tag.append(id_to_tag[id])

    ans = []
    for w, t in zip(word, prediction_tag):
        tuple = {
            'word': w,
            'tag': t
        }
        ans.append(tuple)
    print(ans)
