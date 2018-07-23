def get_data(mode='one_hot'):
    """从指定文件中获得待训练数据，数据源文件是txt文件以', '分割
    PARA:
    filename：数据源文件
    mode：返回值的类型，有one_hot与sequence两种
    RETURN:
    分割好的训练集、测验集
    """
    from sklearn.model_selection import train_test_split
    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences
    from keras.utils import to_categorical
    import pandas as pd
    import numpy as np
    import json
    print("getting data......")
    columns = ['content', 'label']
    content, label = [], []
    with open('D:/instruments_generate/biLstmWithAttention/data/traffic/train.json', mode='r', encoding='utf8') as fp:
        for line in fp.readlines():
            try:
                data_dict = json.loads(line)
                content.append(data_dict['charge']+data_dict['defense']+data_dict['support'])
                label.append(seq2lab(data_dict['result']))
            except:
                pass
    label = to_categorical(np.array(label))
    MAX_LEN = 500
    train_data, test_data, train_label, test_label = train_test_split(content, label,
                                                                      test_size=0.1, random_state=42)
    tokenizer = Tokenizer(
        filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=" ")
    tokenizer.fit_on_texts(content)
    vocab = tokenizer.word_index

    train_data_ids = tokenizer.texts_to_sequences(train_data)
    test_data_ids = tokenizer.texts_to_sequences(test_data)
    if mode == 'one_hot':
        train_data = tokenizer.sequences_to_matrix(
            train_data_ids, mode='binary')
        test_data = tokenizer.sequences_to_matrix(test_data_ids, mode='binary')
    elif mode == 'sequence':
        train_data = pad_sequences(train_data_ids, maxlen=MAX_LEN)
        test_data = pad_sequences(test_data_ids, maxlen=MAX_LEN)
    print("data getted")
    return train_data, test_data, train_label, test_label, vocab


def seq2lab(seq):
    label = 0
    if seq.find('缓刑') >= 0:
        seq = seq[:seq.find('缓刑')]
    if seq.find('一年') >= 0:
        label += 12
    elif seq.find('二年') >= 0:
        label += 24
    elif seq.find('三年') >= 0:
        label += 36
    elif seq.find('四年') >= 0:
        label += 48
    elif seq.find('五年') >= 0:
        label += 60
    elif seq.find('六年') >= 0:
        label += 72

    if seq.find('一月') >= 0:
        label += 1
    elif seq.find('二月') >= 0 or seq.find('二个月') >= 0:
        label += 2
    elif seq.find('三月') >= 0 or seq.find('三个月') >= 0:
        label += 3
    elif seq.find('四月') >= 0 or seq.find('四个月') >= 0:
        label += 4
    elif seq.find('五月') >= 0 or seq.find('五个月') >= 0:
        label += 5
    elif seq.find('六月') >= 0 or seq.find('六个月') >= 0:
        label += 6
    elif seq.find('七月') >= 0 or seq.find('七个月') >= 0:
        label += 7
    elif seq.find('八月') >= 0 or seq.find('八个月') >= 0:
        label += 8
    elif seq.find('九月') >= 0 or seq.find('九个月') >= 0:
        label += 9
    elif seq.find('十月') >= 0 or seq.find('十个月') >= 0:
        label += 10
    elif seq.find('十一月') >= 0 or seq.find('十一个月') >= 0:
        label += 11
    return label


def train_model(train_data, test_data, train_label, test_label, vocab, embedding=200, max_len=500, valid_rate=0.2, drop_out=0.3, batch_size=64, epoch=2):
    """this part is based on lstm"""
    from keras.layers import Dense, Dropout
    from keras.layers import LSTM, Embedding, Bidirectional
    from keras.models import Sequential

    segmentation = int(len(train_data)*valid_rate)
    valid_data = train_data[:segmentation]
    valid_label = train_label[:segmentation]
    train_data = train_data[segmentation+1:]
    train_label = train_label[segmentation+1:]

    model = Sequential()
    model.add(Embedding(len(vocab) + 1, embedding, input_length=max_len))
    model.add(Bidirectional(LSTM(500, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(500)))
    model.add(Dense(len(test_label[0]), activation='softmax'))
    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])

    model.fit(train_data, train_label,
              validation_data=(valid_data, valid_label),
              batch_size=batch_size, epochs=epoch)
    model.save('lstm.h5')
    accuracy = model.evaluate(test_data, test_label)
    print(accuracy)
    date = 'lstm model, embedding = ' + str(embedding)+', max_len='+str(max_len)+', drop_out='+str(drop_out)+', valid_rate='+str(valid_rate) +\
        ', batch_size'+str(batch_size)+', epoch='+str(epoch) + \
        ', accuracy=' + str(accuracy[1])+'\n'
    print(date)
    return accuracy[1]


def main():
    train_data, test_data, train_label, test_label, vocab = get_data(
         mode='sequence')
    train_model(train_data, test_data,
                train_label, test_label, vocab, epoch=5)


if __name__ == '__main__':
    main()
