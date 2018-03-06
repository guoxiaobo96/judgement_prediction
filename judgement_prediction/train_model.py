def get_data_one_hot(filename='D:/judgement_prediction/judgement_prediction/temp/data.txt', mode='one_hot'):
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
    import pandas as pd
    import numpy as np
    
    columns=['content', 'label']
    data = pd.read_csv(filename, encoding='utf-8', sep=', ', header=None, names=columns)
    data.reindex(np.random.permutation(data.index))
    content = data['content']
    label = data['label']
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
    return train_data, test_data, train_label, test_label

def main():
    train_data, test_data, train_label, test_label = get_data_one_hot(mode='sequence')
    print(train_data)


if __name__ == '__main__':
    main()