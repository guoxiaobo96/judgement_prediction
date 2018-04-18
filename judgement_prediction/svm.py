#coding:utf-8
def loadText(case_type):
    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer  
    print ('(1) load texts...')
    train_texts = open('D:/judgement_prediction/judgement_prediction/'+case_type+'/train_content.txt', encoding='utf-8').read().split('\n')
    train_labels = open('D:/judgement_prediction/judgement_prediction/'+case_type+'/train_label.txt', encoding='utf-8').read().split('\n')
    test_texts = open('D:/judgement_prediction/judgement_prediction/'+case_type+'/test_content.txt', encoding='utf-8').read().split('\n')
    test_labels = open('D:/judgement_prediction/judgement_prediction/'+case_type+'/test_label.txt', encoding='utf-8').read().split('\n')
    all_text = train_texts + test_texts
    print ('(2) doc to var...') 
    count_v0= CountVectorizer() 
    count_v0.fit_transform(all_text)
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
    return x_train,y_train,x_test,y_test

def train(x_train,y_train,x_test,y_test):
    print ('(3) SVM...')
    from sklearn.svm import SVC   
    svclf = SVC(kernel = 'linear') 
    svclf.fit(x_train,y_train)  
    preds = svclf.predict(x_test)
    num = 0
    preds = preds.tolist()
    for i,pred in enumerate(preds):
        if int(pred) == int(y_test[i]):
            num += 1
    test_accuracy=float(num) / len(preds)
    print ('precision_score: '+str(test_accuracy))
    return test_accuracy

def main():
    case_type=input("please input case type:")
    x_train,y_train,x_test,y_test=loadText(case_type)
    train(x_train,y_train,x_test,y_test)

if __name__=='__main__':
    main()