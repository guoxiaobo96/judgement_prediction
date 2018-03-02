import tensorflow as tf
import xlrd
import jieba
import os
import sys
import random

def xls2txt(foldname='D:/案件数据/故意杀人案'):
    """this part is used to change from xls to txt"""
    classification_number_count=[0,0,0,0,0]
    source_foldname = foldname
    source_file_list = os.listdir(source_foldname)
    target_filename= 'D:/judgement_prediction/judgement_prediction/temp'
    train_content_filename = target_filename+"/train_content.txt"
    train_label_filename = target_filename+"/train_label.txt"
    test_content_filename = target_filename+"/test_content.txt"
    test_label_filename = target_filename+"/test_label.txt"
    train_content = str()
    train_label = str()
    test_content = str()
    test_label = str()
    case_number = 0

    for source_filename in source_file_list:
        source_filename = source_foldname+"/"+source_filename
        source_file = xlrd.open_workbook(filename=source_filename,encoding_override='utf-8')
        sheet = source_file.sheet_by_index(0)
        content = str()
        for _ in range(1, sheet.nrows):
            case_number +=1
            content = sheet.cell(_, 13).value+" "+sheet.cell(_, 7).value+sheet.cell(_, 8).value+" "
            position = content.find("本院认为")
            if position>0:
                content = content[position:]
            content = content.replace('\n', '')+'\n'
            temp_label = sheet.cell(_, 14).value+''
            distribute_number = random.random()
            label, classification_number = getLabel(temp_label)
            classification_number_count[classification_number]+=1
            if distribute_number<=0.8:
                train_content+=content
                train_label+=label
            else:
                test_content+=content
                test_label+=label
    delete_list = ['、', '：', '。', '，', '“', '”', '《', '》', '＜', '＞',
                   '（', '）', '[', ']', '【', '】', '*', '-', '；']
    for i in range(len(delete_list)):
        train_content = train_content.replace(delete_list[i], '')
        test_content = test_content.replace(delete_list[i], '')
    train_content = " ".join(jieba.cut(train_content))
    test_content = " ".join(jieba.cut(test_content))
    with open(file=train_content_filename, mode="a",encoding='utf-8') as target_file:
        target_file.write(train_content)
    with open(file=train_label_filename, mode="a",encoding='utf-8') as target_file:
        target_file.write(train_label)
    with open(file=test_content_filename, mode="a",encoding='utf-8') as target_file:
        target_file.write(test_content)
    with open(file=test_label_filename, mode="a",encoding='utf-8') as target_file:
        target_file.write(test_label)
    print("case number:%d"%case_number)
    print("death number:%d"%classification_number_count[1])
    print("life number:%d"%classification_number_count[0])
    return "xls2txt finish"

def getLabel(content):
    """this part is used to label the case and it should be changed when dealing with different cases"""
    if content.find("死刑")==-1:
#        if content.find("缓刑")!=-1:
        return "0\n", 0
#        elif content.find("无期徒刑")!=-1:
#            return "3\n"
#        elif content.find("十")!=-1:
#            return "2\n"
#        else:
#            return "1\n"
    else:
        return "1\n", 1

def load_data(fold_name):
    """get texts and labels"""
    train_texts = open(fold_name+'/train_content.txt', encoding='utf-8').read().split('\n')
    train_labels = open(fold_name+'/train_label.txt', encoding='utf-8').read().split('\n')
    test_texts = open(fold_name+'/test_content.txt', encoding='utf-8').read().split('\n')
    test_labels = open(fold_name+'/test_label.txt', encoding='utf-8').read().split('\n')
    print("load data……")
    return train_texts, train_labels, test_texts, test_labels

def doc2vec(train_texts, train_labels, test_texts, test_labels):
    """convert the doc to vec"""
    print("doc to vec")
    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
    count_v0= CountVectorizer() 
    counts_all = count_v0.fit_transform(train_texts+test_texts)
    counts_train= CountVectorizer(vocabulary=count_v0.vocabulary_).fit_transform(train_texts)   
    print ("the shape of train is "+repr(counts_train.shape))  
    counts_test = CountVectorizer(vocabulary=count_v0.vocabulary_).fit_transform(test_texts)  
    print ("the shape of test is "+repr(counts_test.shape))  
  
    tfidftransformer = TfidfTransformer()
    train_data = tfidftransformer.fit_transform(counts_train)
    test_data = tfidftransformer.fit_transform(counts_test) 

    return train_data, train_labels, test_data, test_labels

def svm(train_data, train_labels, test_data, test_labels):
    """use line svm with line kernal and ovr mode"""
    print("sym……")
    from sklearn.svm import LinearSVC
    import numpy
    svclf = LinearSVC() 
    svm_feature_list=svclf.fit(train_data,train_labels).coef_
    numpy.savetxt("svm.txt", svm_feature_list)
    with open(file="svm.txt", mode="a",encoding='utf-8') as target_file:
        for i, svm_feature in enumerate(svm_feature_list):
            numpy.savetxt("svm"+str(i)+".txt", svm_feature)
    preds = svclf.predict(test_data)
    num = 0
    preds = preds.tolist()
    for i,pred in enumerate(preds):
        if int(pred) == int(test_labels[i]):
            num += 1
    print ('precision_score: '+str(float(num) / len(preds)))

def main():
    xls2txt()

main()