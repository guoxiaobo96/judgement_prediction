class readDate():

    def __init__(self,case_type,data_name):
        self.case_type=case_type
        self.data_name=data_name

    def __read_data(self):
        from keras.utils import to_categorical
        import pandas as pd
        import numpy as np
        print("getting data......")
        columns=['content', 'label']
        filename='D:/judgement_prediction/judgement_prediction/'+self.case_type+'/'+self.data_name
        data = pd.read_csv(filename, encoding='utf-8', sep=', ', header=None, names=columns, engine='python')
        data.reindex(np.random.permutation(data.index))
        content = data['content']
        label = to_categorical(np.array(data['label']))
        return content, label
    
    def __split_data(self,mode,MAX_LEN):
        from sklearn.model_selection import train_test_split
        from keras.preprocessing.text import Tokenizer
        from keras.preprocessing.sequence import pad_sequences
        content, label=self.__read_data()
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
        return train_data, test_data, train_label, test_label, vocab
    
    def process_data(self,mode,MAX_LEN):
        return self.__split_data(mode,MAX_LEN)
