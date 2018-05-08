def xls2txt(case_name,data_type,char_split=True):
    import jieba
    import os
    import xlrd
    """this part is used to change the date from xls version to txt version
    PARA:
    data_type:如果为0则将案件结果简单分为缓刑、轻型、重型、无期和死刑，且放在同一文件中，如果为1则具体到年，但是放在同一文件中，如果为2则具体到年但是根据量刑的轻重放在不同文件中"""
    classification_number_count=[0 for i in range(50)]
    source_foldname = 'D:/案件数据/'+case_name
    source_file_list = os.listdir(source_foldname)
    target_filename= 'D:/judgement_prediction/hierachy/'+case_name+'/'

    for source_filename in source_file_list:
        source_filename = source_foldname+"/"+source_filename
        source_file = xlrd.open_workbook(filename=source_filename,encoding_override='utf-8')
        sheet = source_file.sheet_by_index(0)
        content = str()
        for _ in range(1, sheet.nrows):
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
            delete_list = ['、', '：', '。', '，', '“', '”', '《', '》', '＜', '＞',
                    '（', '）', '[', ']', '【', '】', '*', '-', '；', ',']
            for i in range(len(delete_list)):
                    content = content.replace(delete_list[i], '')
            if data_type==0:
                data=str()
                label, classification_number = getFirstLabel(temp_label)
                classification_number_count[classification_number]+=1
                data=data+content+','+label
                if char_split==True:
                    data= " ".join(jieba.cut(data))
                data=data.replace(' ','')
                with open(file=target_filename+'data.txt', mode="a",encoding='utf-8') as target_file:
                    target_file.write(data)
            
            elif data_type==1:
                data=str()
                label, classification_number = getSecondLabel(temp_label)
                classification_number_count[classification_number]+=1
                data=data+content+','+label
                if char_split==True:
                    data= " ".join(jieba.cut(data))
                with open(file=target_filename+'data.txt', mode="a",encoding='utf-8') as target_file:
                    target_file.write(data)

            elif data_type==2:
                data_imprison=str()
                data_misdemeanor=str()
                data_felony=str()
                data_life_long=str()
                data_death=str()
                label, classification_number = getSecondLabel(temp_label)
                classification_number_count[classification_number]+=1
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
                if char_split==True:
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
    return "xls2txt finish"

def getSecondLabel(content):
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

def getFirstLabel(label):
    if label.find('死刑')!=-1:
        if label.find('缓刑')!=-1:
            return "0\n", 0
        elif label.find('徒刑十')==-1:
            return '1\n', 1
        elif label.find('无期')!=-1:
            return '3\n', 3
        else:
            return '2\n', 2
    else:
        return "4\n", 4

def main():
    case_name=input('please input case name:')
    data_type=int(input('please input data_type:'))
    xls2txt(case_name,data_type)

if __name__=='__main__':
    main()