import os
import sys
import random
import xlrd
import jieba

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
            content = sheet.cell(_, 13).value
            position = content.find("本院认为")
            if position>0:
                content = content[position:]
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
    print("death number:%d"%classification_number_count[1])
    print("life number:%d"%classification_number_count[0])
    return "xls2txt finish"

def xls2csv(foldname='D:/案件数据/故意杀人案'):
    """this part is used to change the date from xls version to csv version"""
    classification_number_count=[0,0,0,0,0]
    source_foldname = foldname
    source_file_list = os.listdir(source_foldname)
    target_filename= 'D:/judgement_prediction/judgement_prediction/temp/data.txt'
    
    data=str()
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
            content = sheet.cell(_, 7).value+" "+sheet.cell(_, 8).value+" "+sheet.cell(_, 11).value[0:4]+" "+content
            content = content.replace('\n', '')
            temp_label = sheet.cell(_, 14).value+''
            label, classification_number = getLabel(temp_label)
            classification_number_count[classification_number]+=1
            delete_list = ['、', '：', '。', '，', '“', '”', '《', '》', '＜', '＞',
                   '（', '）', '[', ']', '【', '】', '*', '-', '；', ',']
            for i in range(len(delete_list)):
                content = content.replace(delete_list[i], '')
            data = data+content+","+label[0:1]+'\n'
    data = " ".join(jieba.cut(data))
    with open(file=target_filename, mode="a",encoding='utf-8') as target_file:
        target_file.write(data)
    print("case number:%d"%case_number)
    print("death number:%d"%classification_number_count[1])
    print("life number:%d"%classification_number_count[0])
    return "xls2csv finish"

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

def main():
    """主程序，根据输入决定操作内容。
    prepare: 对于文本进行适当处理
    cmm: 采用textcnn对于处理好的文本进行分类"""
    value = str('')
    value = input("please input action:")
    while value != 'quit':
        if value == 'txt prepare':
            xls2txt()
        elif value=='csv prepare':
            xls2csv()
        value = input("please input action:")

main()