import random
import numpy as np
# can be sentence or word


def get_data_raw(file_name):
    data = []
    with open(file=file_name, mode='r', encoding='utf8') as file:
        for line in file:
            if line.find(',') == -1:
                continue
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
    content_id, label_id = [], []
    document_len, sentence_len = 0, 0
    for _, content in enumerate(raw_data):
        document_id = []
        label_id.append(content[1])
        if len(content[0]) > document_len:
            document_len = len(content[0])
        for _, sentence in enumerate(content[0]):
            if len(sentence) > 100:
                continue
            if len(sentence) > sentence_len:
                sentence_len = len(sentence)
            sentence_id = [word_to_id[x]
                           for _, x in enumerate(sentence) if x in word_to_id]
            document_id.append(sentence_id)
        content_id.append(document_id)

    x_data = content_id
    y_data = label_id
    data_size = len(x_data)
    x_train = x_data[:int(train_rate*data_size)]
    y_train = y_data[:int(train_rate*data_size)]
    x_val = x_data[int(train_rate*data_size)+1:int((1-test_rate)*data_size)]
    y_val = y_data[int(train_rate*data_size)+1:int((1-test_rate)*data_size)]
    x_test = x_data[int((1-test_rate)*data_size)+1:]
    y_test = y_data[int((1-test_rate)*data_size)+1:]
    if not split:
        return x_data, y_data
    return x_train, y_train, x_val, y_val, x_test, y_test, document_len, sentence_len


def get_question(question_dir, word_to_id):
    question_id, question = [], []
    question_len = 0
    with open(file=question_dir, mode='r', encoding='utf8') as file:
        for line in file:
            question.append(line.strip())
    for _, sentence in enumerate(question):
        if len(sentence) > question_len:
            question_len = len(sentence)
        question_id=([word_to_id[x]
                            for _, x in enumerate(sentence) if x in word_to_id])
    return question_id, question_len


def load_data(config):
    raw_data = get_data_raw('D:/criminal_data/criminal/data.txt')
    words, word_to_id = read_vocab('d:/criminal_data/criminal/vocab.txt')
    x_train, y_train, x_val, y_val, x_test, y_test, document_len, sentence_len = process_data(
        raw_data, word_to_id)
    questions, question_len = get_question(
        'D:/criminal_data/criminal/question.txt', word_to_id)
    if not config.word2vec_init:
        word_embedding = np.random.uniform(-config.embedding_init,
                                           config.embedding_init, (len(words), config.embedding_size))
    return x_train, y_train, x_val, y_val, x_test, y_test, questions, word_embedding, question_len, document_len, sentence_len, len(words), len(questions)

if __name__=='__main__':
    raw_data = get_data_raw('D:/criminal_data/criminal/data.txt')
    words, word_to_id = read_vocab('d:/criminal_data/criminal/vocab.txt')
    x_train, y_train, x_val, y_val, x_test, y_test, document_len, sentence_len = process_data(
        raw_data, word_to_id)
    questions, question_len = get_question(
        'D:/criminal_data/criminal/question.txt', word_to_id)