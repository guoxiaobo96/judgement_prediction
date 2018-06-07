import os
import random
import numpy as np
import keras

# can be sentence or word
INPUT_MASK_MODE = "sentence"


def get_data_raw(file_name):
    data = []
    with open(file=file_name, mode='r', encoding='utf8') as file:
        for line in file:
            content, label = line.strip().split(',')
            if content:
                data.append([content.split('。'), label])
    return data


def read_vocab(vocab_dir):
    with open(vocab_dir, mode='r', encoding='utf8') as vocab_file:
        words = [_.strip() for _ in vocab_file.readlines()]
        word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id


def process_data(raw_data, word_to_id, train_rate=0.6, test_rate=0.2, split=True):
    random.shuffle(raw_data)
    content_id, label_id, document_pad = [], [], []
    document_len, sentence_len = 0, 0
    for _, content in enumerate(raw_data):
        document_id = []
        label_id = content[1]
        if len(content[0]) > document_len:
            document_len = len(content[0])
        for _, sentence in enumerate(content[0]):
            if len(content[0]) > sentence_len:
                sentence = len(content[0])
            sentence_id = []
            sentence_id.append([word_to_id[x]
                                for x in sentence if x in word_to_id])
            document_id.append(sentence_id)
        content_id.append(document_id)

    for _, document in enumerate(content_id):
        document_pad.append(
            keras.preprocessing.sequence.pad_sequences(document, sentence_len))
    x_data = keras.preprocessing.sequence.pad_sequences(
        document_pad, document_len)
    y_data = keras.utils.to_categorical(label_id)
    data_size = len(x_data)
    x_train = x_data[:int(train_rate*data_size)]
    y_train = y_data[:int(train_rate*data_size)]
    x_val = x_data[int(train_rate*data_size)+1:int((1-test_rate)*data_size)]
    y_val = y_data[int(train_rate*data_size)+1:int((1-test_rate)*data_size)]
    x_test = x_data[int((1-test_rate)*data_size)+1:]
    y_test = y_data[int((1-test_rate)*data_size)+1:]
    if not split:
        return x_data, y_data
    return x_train, y_train, x_val, y_val, x_test, y_test


def main():
    get_data_raw('D:/criminal_data/criminal/data.txt')


if __name__ == '__main__':
    main()
