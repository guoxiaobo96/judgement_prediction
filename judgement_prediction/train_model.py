def get_data_one_hot(filename='D:/judgement_prediction/judgement_prediction/temp/data.txt'):
    from sklearn.model_selection import train_test_split
    from keras.preprocessing.text import Tokenizer
    import pandas as pd
    columns=['content', 'label']
    data = pd.read_csv(filename, encoding='utf-8', sep=', ', header=None, names=columns)
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

    train_data = tokenizer.sequences_to_matrix(train_data_ids, mode='binary')
    test_data = tokenizer.sequences_to_matrix(test_data_ids, mode='binary')
    return train_data, test_data, train_label, test_label

def textcnn_model(foldname='D:/judgement_prediction/judgement_prediction/temp/data.csv'):
    return 0