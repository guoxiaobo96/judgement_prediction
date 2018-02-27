#coding:utf-8
import sys

VECTOR_DIR = 'vectors.bin'

MAX_SEQUENCE_LENGTH = 200
EMBEDDING_DIM = 200
TEST_SPLIT = 0.2


print ('(1) load texts...')
train_texts = open('D:/案件数据/故意杀人案/train_content.txt', encoding='utf-8').read().split('\n')
train_labels = open('D:/案件数据/故意杀人案/train_label.txt', encoding='utf-8').read().split('\n')
test_texts = open('D:/案件数据/故意杀人案/test_content.txt', encoding='utf-8').read().split('\n')
test_labels = open('D:/案件数据/故意杀人案/test_label.txt', encoding='utf-8').read().split('\n')
all_text = train_texts + test_texts

print ('(2) doc to var...')
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer   
count_v0= CountVectorizer() 
counts_all = count_v0.fit_transform(all_text)
count_v1= CountVectorizer(vocabulary=count_v0.vocabulary_)
counts_train = count_v1.fit_transform(train_texts)
print ("the shape of train is "+repr(counts_train.shape))  
count_v2 = CountVectorizer(vocabulary=count_v0.vocabulary_)
counts_test = count_v2.fit_transform(test_texts)
print ("the shape of test is "+repr(counts_test.shape))  
  
tfidftransformer = TfidfTransformer()
train_data = tfidftransformer.fit(counts_train).transform(counts_train)
test_data = tfidftransformer.fit(counts_test).transform(counts_test)

x_train = train_data
y_train = train_labels
x_test = test_data
y_test = test_labels

print ('(3) KNN...')
from sklearn.neighbors import KNeighborsClassifier  

for x in range(1,15):  
    knnclf = KNeighborsClassifier(n_neighbors=x)
    knnclf.fit(x_train,y_train)  
    preds = knnclf.predict(x_test)
    num = 0
    preds = preds.tolist()
    for i,pred in enumerate(preds):
        if int(pred) == int(y_test[i]):
            num += 1
    print ('K= '+str(x)+', precision_score:' + str(float(num) / len(preds)))





        




