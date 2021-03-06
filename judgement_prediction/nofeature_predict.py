import os
import sys
import random
import xlrd
import jieba

def xls2txt(case_name):
    """this part is used to change from xls to txt"""
    classification_number_count=[0,0,0,0,0]
    source_foldname = 'D:/案件数据/'+case_name
    source_file_list = os.listdir(source_foldname)
    target_filename= 'D:/judgement_prediction/judgement_prediction/'+case_name
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
            content = sheet.cell(_, 13).value
            position = content.find("本院认为")
            if position>0:
                content = content[position:]
            position = content.find("中华人民共和国刑法")
            if position>0:
                content = content[:position]
            content = sheet.cell(_, 7).value+" "+sheet.cell(_, 8).value+" "+sheet.cell(_, 11).value[0:4]+" "+content
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
    print("temp number:%d"%classification_number_count[0])
    print("short number:%d"%classification_number_count[1])
    print("long number:%d"%classification_number_count[2])
    print("life long number:%d"%classification_number_count[3])
    print("death number:%d"%classification_number_count[4])
    return "xls2txt finish"

def xls2csv(case_name):
    """this part is used to change the date from xls version to csv version"""
    classification_number_count=[0 for i in range(50)]
    source_foldname = 'D:/案件数据/'+case_name
    source_file_list = os.listdir(source_foldname)
    target_filename= 'D:/judgement_prediction/judgement_prediction/'+case_name+'/'
    
    data_imprison=str()
    data_misdemeanor=str()
    data_felony=str()
    data_life_long=str()
    data_death=str()
    case_number = 0

    for source_filename in source_file_list:
        source_filename = source_foldname+"/"+source_filename
        source_file = xlrd.open_workbook(filename=source_filename,encoding_override='utf-8')
        sheet = source_file.sheet_by_index(0)
        content = str()
        for _ in range(1, sheet.nrows):
            case_number +=1
            content = sheet.cell(_, 13).value
            position = content.find("本院认为")
            if position>0:
                content = content[position:]
            position = content.find("中华人民共和国刑法")
            if position>0:
                content = content[:position]
            content = sheet.cell(_, 7).value+" "+sheet.cell(_, 8).value+" "+sheet.cell(_, 11).value[0:4]+" "+content
            content = content.replace('\n', '')
            temp_label = sheet.cell(_, 14).value+''
            label, classification_number = getLabel(temp_label)
            classification_number_count[classification_number]+=1
            delete_list = ['、', '：', '。', '，', '“', '”', '《', '》', '＜', '＞',
                   '（', '）', '[', ']', '【', '】', '*', '-', '；', ',']
            for i in range(len(delete_list)):
                content = content.replace(delete_list[i], '')
            if int(label)==0:
                data_imprison = data_imprison+content+","+label
            elif int(label)>0 and int(label)<10:
                data_misdemeanor= data_misdemeanor+content+","+label
            elif int(label)>=10 and int(label)<30:
                data_felony= data_felony+content+","+label
            elif int(label)==30:
                data_life_long=data_life_long+content+','+label
            else:
                data_death=data_death+content+','+label
    data_imprison= " ".join(jieba.cut(data_imprison))
    data_misdemeanor= " ".join(jieba.cut(data_misdemeanor))
    data_death = " ".join(jieba.cut(data_death))
    data_felony= " ".join(jieba.cut(data_felony))
    data_life_long=" ".join(jieba.cut(data_life_long))
    with open(file=target_filename+'imprison.txt', mode="w",encoding='utf-8') as target_file:
        target_file.write(data_imprison)
    with open(file=target_filename+'misdemeanor.txt', mode="w",encoding='utf-8') as target_file:
        target_file.write(data_misdemeanor)
    with open(file=target_filename+'felony.txt', mode="w",encoding='utf-8') as target_file:
        target_file.write(data_felony)
    with open(file=target_filename+'life.txt', mode="w",encoding='utf-8') as target_file:
        target_file.write(data_life_long)
    with open(file=target_filename+'death.txt', mode="w",encoding='utf-8') as target_file:
        target_file.write(data_death)
    return "xls2csv finish"

def getLabel(content):
    """this part is used to label the case and it should be changed when dealing with different cases"""
    if content.find("死刑")==-1:
        if content.find("缓刑")!=-1:
            return "0\n", 0
        elif content.find("无期徒刑")!=-1:
            return "30\n", 30
        elif content.find("徒刑十")!=-1:
            if content.find("徒刑十一年")!=-1:
                return "11\n", 11
            elif content.find("徒刑十二年")!=-1:
                return "12\n", 12
            elif content.find("徒刑十三年")!=-1:
                return "13\n", 13
            elif content.find("徒刑十四年")!=-1:
                return "14\n", 14
            elif content.find("徒刑十五年")!=-1:
                return "15\n", 15
            elif content.find("徒刑十六年")!=-1:
                return "16\n", 16
            elif content.find("徒刑十七年")!=-1:
                return "17\n", 17
            elif content.find("徒刑十八年")!=-1:
                return "18\n", 18
            elif content.find("徒刑十九年")!=-1:
                return "19\n", 19
            else:
                return "10\n",10
        elif content.find("徒刑二十")!=-1:
            if content.find("徒刑二十一年")!=-1:
                return "21\n", 21
            elif content.find("徒刑二十二年")!=-1:
                return "22\n", 22
            elif content.find("徒刑二十三年")!=-1:
                return "23\n", 23
            elif content.find("徒刑二十四年")!=-1:
                return "24\n", 24
            elif content.find("徒刑二十五年")!=-1:
                return "25\n", 25
            else:
                return "20\n",20
        else:
            if content.find("刑一年")!=-1:
                return "1\n", 1
            elif content.find("刑二年")!=-1:
                return "2\n", 2
            elif content.find("刑三年")!=-1:
                return "3\n", 3
            elif content.find("刑四年")!=-1:
                return "4\n", 4
            elif content.find("刑五年")!=-1:
                return "5\n", 5
            elif content.find("刑六年")!=-1:
                return "6\n", 6
            elif content.find("刑七年")!=-1:
                return "7\n", 7
            elif content.find("刑八年")!=-1:
                return "8\n", 8
            elif content.find("刑九年")!=-1:
                return "9\n", 9
            else:
                return "10\n",10
    else:
        return "40\n", 40

def train(model,case_type,number=1):
    average_accuracy=0
    test_accuracy=list()
    if model=='svm':
        import svm
        x_train,y_train,x_test,y_test=svm.loadText(case_type)
        for _ in range(int(number)):
            test_accuracy.append(svm.train(x_train,y_train,x_test,y_test)) 
            average_accuracy=average_accuracy+test_accuracy[_]
        average_accuracy=average_accuracy/int(number)
        print("aveage accuract:" +str(average_accuracy))
    elif model=='cnn':
        import cnn
        train_data, test_data, train_label, test_label, vocab = cnn.get_data(case_type,mode='sequence')
        for _ in range(int(number)):
            test_accuracy.append(cnn.train_model(case_type,train_data, test_data, train_label, test_label, vocab))
            average_accuracy=average_accuracy+test_accuracy[_]
    elif model=='lstm':
        import lstm
        train_data, test_data, train_label, test_label, vocab = lstm.get_data(case_type,mode='sequence')
        for _ in range(int(number)):
            test_accuracy.append(lstm.train_model(case_type,train_data, test_data, train_label, test_label, vocab))
            average_accuracy=average_accuracy+test_accuracy[_]
    elif model=='keras_text_cnn':
        import keras_text_cnn as text_cnn
        train_data, test_data, train_label, test_label, vocab = text_cnn.get_data(case_type,mode='sequence')
        for _ in range(int(number)):
            test_accuracy.append(text_cnn.train_model(case_type,train_data, test_data, train_label, test_label, vocab))
            average_accuracy=average_accuracy+test_accuracy[_]
    average_accuracy=average_accuracy/int(number)
    print("aveage accuract:" +str(average_accuracy))
    with open(file='D:/judgement_prediction/judgement_prediction/'+case_type+'/information.txt', mode="a",encoding='utf-8') as target_file:
        target_file.write(case_type)
        for i in range(int(number)):
            target_file.write(str(test_accuracy[i])+' ')
        target_file.write(',average:'+str(average_accuracy)+'\n')
            
def main():
    """主程序，根据输入决定操作内容。
    prepare: 对于文本进行适当处理
    cmm: 采用textcnn对于处理好的文本进行分类"""
    command_line = str('')
    command_line = input("please input action:")
    while command_line != 'quit':
        if command_line.split()[0]=='prepare':
            if command_line.split()[1] == 'txt':
                xls2txt(command_line.split()[2])
            elif command_line.split()[1]=='csv':
                xls2csv(command_line.split()[2])
        elif command_line.split()[0]=='train':
            train(command_line.split()[1],command_line.split()[2],command_line.split()[3])
        command_line = input("please input action:")

main()