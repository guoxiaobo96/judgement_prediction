def get_data(case_type, mode='one_hot'):
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
    filename='D:/judgement_prediction/judgement_prediction/'+case_type+'/data.txt'
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

def train_model(case_type,train_data, test_data, train_label, test_label, vocab,embedding = 200, max_len = 200, valid_rate = 0.2, drop_out=0.3, batch_size =64, epoch=2):
    """this part is based on lstm"""
    from keras.layers import Dense, Dropout
    from keras.layers import LSTM, Embedding
    from keras.models import Sequential

    segmentation = int(len(train_data)*valid_rate)
    valid_data = train_data[:segmentation]
    valid_label = train_label[:segmentation]
    train_data = train_data[segmentation+1:]
    train_label = train_label[segmentation+1:]

    model = Sequential()
    model.add(Embedding(len(vocab) + 1, embedding, input_length=max_len))
    model.add(LSTM(200, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dropout(0.2))
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
    date = 'lstm model, embedding = '+ str(embedding)+', max_len='+str(max_len)+', drop_out='+str(drop_out)+', valid_rate='+str(valid_rate)+\
            ', batch_size'+str(batch_size)+', epoch='+str(epoch)+', accuracy='+ str(accuracy[1])+'\n'
    with open(file='D:/judgement_prediction/judgement_prediction/'+case_type+'/information.txt', mode="a",encoding='utf-8') as target_file:
        target_file.write(date)
    return accuracy[1]

def main():
    case_type=input("please input case type:")
    train_data, test_data, train_label, test_label, vocab = get_data(case_type,mode='sequence')
    train_model(case_type,train_data, test_data, train_label, test_label, vocab,epoch=5)

if __name__=='__main__':
    main()