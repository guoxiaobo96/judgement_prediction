def get_data(filename='D:/judgement_prediction/judgement_prediction/temp/data.txt', mode='one_hot'):
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
    print("getting data......")
    columns=['content', 'label']
    data = pd.read_csv(filename, encoding='utf-8', sep=', ', header=None, names=columns, engine='python')
    data.reindex(np.random.permutation(data.index))
    content = data['content']
    label = to_categorical(np.array(data['label']))
    MAX_LEN = 200
    train_data, test_data, train_label, test_label = train_test_split(content, label,
                                                                      test_size=0.1, random_state=42)
    tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',lower=True,split=" ")
    tokenizer.fit_on_texts(content)
    vocab = tokenizer.word_index

    train_data_ids = tokenizer.texts_to_sequences(train_data)
    test_data_ids = tokenizer.texts_to_sequences(test_data)
    if mode=='one_hot':
        train_data = tokenizer.sequences_to_matrix(train_data_ids, mode='binary')
        test_data = tokenizer.sequences_to_matrix(test_data_ids, mode='binary')
    elif mode=='sequence':
        train_data = pad_sequences(train_data_ids, maxlen=MAX_LEN)
        test_data = pad_sequences(test_data_ids, maxlen=MAX_LEN)
    print("data getted")
    return train_data, test_data, train_label, test_label, vocab


def cnn_model(embedding = 200, max_len = 200, valid_rate = 0.5, drop_out=0.3, batch_size =64, epoch=3):
    """this part is based on cnn"""
    from keras.layers import Dense, Flatten, Dropout
    from keras.layers import Conv1D, MaxPooling1D, Embedding, Convolution1D, BatchNormalization
    from keras.models import Sequential

    train_data, test_data, train_label, test_label, vocab = get_data(mode='sequence')
    segmentation = int(len(train_data)*valid_rate)
    valid_data = train_data[:segmentation]
    valid_label = train_label[:segmentation]
    train_data = train_data[segmentation+1:]
    train_label = train_label[segmentation+1:]

    print("cnn......")
    model = Sequential()
    model.add(Embedding(len(vocab)+1, embedding, input_length=max_len))
    model.add(Convolution1D(256, 3, padding = 'same'))
    model.add(MaxPooling1D(3, 3, padding='same'))
    model.add(Convolution1D(128, 3, padding = 'same'))
    model.add(MaxPooling1D(3, 3, padding='same'))
    model.add(Convolution1D(64, 3, padding = 'same'))
    model.add(Flatten())
    model.add(Dropout(drop_out))
    model.add(BatchNormalization())
    model.add(Dense(256, activation = 'relu'))
    model.add(Dense(embedding, activation='relu'))
    model.add(Dropout(drop_out))
    model.add(Dense(2, activation='softmax'))
    model.summary()

    model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

    model.fit(train_data, train_label,
              validation_data=(valid_data, valid_label),
              batch_size=batch_size, epochs=epoch)
    accuracy = model.evaluate(test_data, test_label)
    print(accuracy)
    date = 'cnn model, embedding = '+ str(embedding)+', max_len='+str(max_len)+', drop_out='+str(drop_out)+', valid_rate='+str(valid_rate)+\
            ', batch_size'+str(batch_size)+', epoch='+str(epoch)+', accuracy='+ str(accuracy[1])+'\n'
    with open(file='D:/judgement_prediction/judgement_prediction/temp/information.txt', mode="a",encoding='utf-8') as target_file:
        target_file.write(date)
    return accuracy[1]

def main():
    cnn_model(embedding=200, max_len=200,drop_out=0.15)

if __name__ == '__main__':
    main()