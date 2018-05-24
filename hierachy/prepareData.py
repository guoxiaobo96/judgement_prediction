class DealRawData():
    def __init__(self,case_name):
        self.case_name=case_name
        self.data_list=[]
        self.target_filename= 'D:/judgement_prediction/hierachy/'+self.case_name+'/'
        self.classification_number=41
        for _ in range(self.classification_number):
                self.data_list.append('')
        self.min_frequence=2
        

    def xls2txt(self,data_type,char_split=False):
        import jieba
        import xlrd
        """this part is used to change the date from xls version to txt version
        PARA:
        data_type:如果为0则将案件结果简单分为缓刑、轻型、重型、无期和死刑，且放在同一文件中，如果为1则具体到年，但是放在同一文件中，如果为2则具体到年但是根据量刑的轻重放在不同文件中"""
        classification_number_count=[0 for i in range(self.classification_number)]
        source_foldname = 'D:/案件数据/'+self.case_name
        source_filename=source_foldname+'/all_data.xls'

        source_file = xlrd.open_workbook(filename=source_filename,encoding_override='utf-8')
        sheet = source_file.sheet_by_index(0)
        content = str()
        data=str()
        data_imprison=str()
        data_misdemeanor=str()
        data_felony=str()
        data_life_long=str()
        data_death=str()

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
                label, classification_number = self.__getFirstLabel(temp_label)
                classification_number_count[classification_number]+=1
                self.data_list[classification_number]=self.data_list[classification_number]+content+','+label
            
            elif data_type==1:
                label, classification_number = self.__getSecondLabel(temp_label)
                classification_number_count[classification_number]+=1
                self.data_list[classification_number]=self.data_list[classification_number]+content+','+label

            elif data_type==2:
                label, classification_number = self.__getSecondLabel(temp_label)
                classification_number_count[classification_number]+=1
                temp_place=label.find(',')
                if int(label[temp_place+1:])==0:
                    data_imprison = data_imprison+content+","+label
                elif int(label[temp_place+1:])>0 and int(label[temp_place+1:])<10:
                    data_misdemeanor= data_misdemeanor+content+","+label
                elif int(label[temp_place+1:])>=10 and int(label[temp_place+1:])<30:
                    data_felony= data_felony+content+","+label
                elif int(label[temp_place+1:])==30:
                    data_life_long=data_life_long+content+','+label
                else:
                    data_death=data_death+content+','+label
                self.data_list[classification_number]=self.data_list[classification_number]+content+','+label[temp_place+1:]
            
        if data_type==0 or data_type==1:
            if char_split==True:
                for i in range(self.classification_number):
                    self.data_list[i]= " ".join(jieba.cut(self.data_list[i]))
                    self.data_list[i]=self.data_list[i].replace('  ',' ')

            
        elif data_type==2:
            if char_split==True:
                data_imprison= " ".join(jieba.cut(data_imprison))
                data_imprison=data_imprison.replace('  ',' ')
                data_misdemeanor= " ".join(jieba.cut(data_misdemeanor))
                data_misdemeanor=data_misdemeanor.replace('  ',' ')
                data_death = " ".join(jieba.cut(data_death))
                data_death=data_death.replace('  ',' ')
                data_felony= " ".join(jieba.cut(data_felony))
                data_felony=data_felony.replace('  ',' ')
                data_life_long=" ".join(jieba.cut(data_life_long))
                data_life_long=data_life_long.replace('  ',' ')

            with open(file=self.target_filename+'imprison.txt', mode="a",encoding='utf-8') as target_file:
                target_file.write(data_imprison)
            with open(file=self.target_filename+'misdemeanor.txt', mode="a",encoding='utf-8') as target_file:
                target_file.write(data_misdemeanor)
            with open(file=self.target_filename+'felony.txt', mode="a",encoding='utf-8') as target_file:
                target_file.write(data_felony)
            with open(file=self.target_filename+'life.txt', mode="a",encoding='utf-8') as target_file:
                target_file.write(data_life_long)
            with open(file=self.target_filename+'death.txt', mode="a",encoding='utf-8') as target_file:
                target_file.write(data_death)
        min_number=9999
        for _, number in enumerate(classification_number_count):
            if number<min_number and number>0:
                min_number=number
        self.__filterAndConverge(min_number)

        if(data_type!=2):
            build_vocab(train_data=self.target_filename+'data.txt',vocab_dir=self.target_filename+'vocab.txt',vocab_size=5000,min_frequence=1)
        return "xls2txt finish"

    def __getSecondLabel(self,content):
        """this part is used to label the case and it should be changed when dealing with different cases"""
        content=content.replace('\n','')
        content=content.replace(',','，')
        if content.find("死刑")==-1:
            if content.find("缓刑")!=-1:
                return "0\n", 0
            elif content.find("无期徒刑")!=-1:
                return "21\n", 21
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
            return "22\n", 22

    def __getFirstLabel(self,label):
        if label.find('死刑')==-1:
            if label.find('缓刑')!=-1:
                return "0\n", 0
            elif label.find('徒刑十')!=-1:
                return '1\n', 2
            elif label.find('无期')!=-1:
                return '3\n', 3
            else:
                return '2\n', 1
        else:
            return "4\n", 4

    def __filterAndConverge(self,min_number):
        import random
        all_data=str()
        data_for_model=str()
        for i, data in enumerate(self.data_list):
            temp_list=data.split('\n')
            random.shuffle(temp_list)
            with open(file=self.target_filename+str(i)+'.txt', mode="w",encoding='utf-8') as target_file:
                    target_file.write(data)
            all_data+=data
            for j in range(min_number):
                temp_list[j]+='\n'
                data_for_model+=temp_list[j]
        with open(file=self.target_filename+'data.txt', mode="w",encoding='utf-8') as target_file:
            target_file.write(all_data)
        with open(file=self.target_filename+'data_for_train.txt', mode="w",encoding='utf-8') as target_file:
            target_file.write(data_for_model)




def read_data(file_name):
    contents,labels=[],[]
    with open(file=file_name,mode='r',encoding='utf8') as f:
        for line in f:
            try:
                content, label=line.strip().split(',')
                if content:
                    content=content.replace(' ','')
                    contents.append(content)
                    labels.append(label)
            except:
                    pass
    return contents, labels

def build_vocab(train_data,vocab_dir,vocab_size,min_frequence):
        from collections import Counter
        contents,_=read_data(train_data)
        all_data=[]
        for content in contents:
            all_data.extend(content)
        
        counter = Counter(all_data)
        count_pairs = counter.most_common(vocab_size-1)
        vocab_list=list(zip(*count_pairs))
        index=vocab_list[1].index(1)
        words=vocab_list[0][:index]
        # 添加一个 <PAD> 来将所有文本pad为同一长度
        words = ['<PAD>'] + list(words)
        with open(vocab_dir,mode='w',encoding='utf8') as f:
            f.write('\n'.join(words) + '\n')

def read_vocab(vocab_dir):
    with open(vocab_dir,mode='r',encoding='utf8') as fp:
        words = [_.strip() for _ in fp.readlines()]
        word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id

def batch_iter(x,y, batch_size, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    import numpy as np
    data_size = len(x)
    num_batches_per_epoch = int((data_size-1)/batch_size) + 1
    # Shuffle the data at each epoch
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        x_shuffle = x[shuffle_indices]
        y_shuffle = y[shuffle_indices]
    else:
        x_shuffle=x
        y_shuffle=y
    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        yield x_shuffle[start_index:end_index],y_shuffle[start_index:end_index]

def process_data(file_name, word_to_id, cat_to_id,max_length=600):
    import keras
    import random
    contents,labels=read_data(file_name)
    data_id, label_id,all_data=[],[],[]
    for i in range(len(contents)):
        all_data.append(contents[i]+','+labels[i])
    random.shuffle(all_data)
    contents,labels=[],[]
    for _,data in enumerate(all_data):
        content,label=str(data).strip().split(',')
        contents.append(content)
        labels.append(label)
    for i in range(len(contents)):
        data_id.append([word_to_id[x]for x in contents[i] if x in word_to_id])
        label_id.append(labels[i])
    x_pad=keras.preprocessing.sequence.pad_sequences(data_id,max_length)
    y_pad=keras.utils.to_categorical(label_id)

    return x_pad,y_pad,len(all_data)

def get_data(data_dir,word_to_id,cat_to_id,seq_length,train_rate=0.6,test_rate=0.2):
    x_data,y_data,data_size=process_data(data_dir, word_to_id, cat_to_id, seq_length)
    x_train=x_data[:int(train_rate*data_size)]
    y_train=y_data[:int(train_rate*data_size)]
    x_val=x_data[int(train_rate*data_size)+1:int((1-test_rate)*data_size)]
    y_val=y_data[int(train_rate*data_size)+1:int((1-test_rate)*data_size)]
    x_test=x_data[int((1-test_rate)*data_size)+1:]
    y_test=y_data[int((1-test_rate)*data_size)+1:]
    return x_train,y_train,x_val,y_val,x_test,y_test

def to_words(content,words):
    return ' '.join(words[x] for x in content)

def read_catagory():
    catagories=['1','2','3','4','5','6','7','8','9']
    catagories=[x for x in catagories]
    cat_to_id=dict(zip(catagories,range(len(catagories))))
    return catagories,cat_to_id

def claen_data():
    import xlrd,xlwt
    import os
    source_foldname = 'D:/案件数据/murder_hierachy'
    source_file_list = os.listdir(source_foldname)
    target_file=xlwt.Workbook(encoding='utf-8')
    target_sheet=target_file.add_sheet('sheet1')
    target_col=0
    for source in source_file_list:
        source_filename = source_foldname+"/"+source
        source_file = xlrd.open_workbook(filename=source_filename,encoding_override='utf-8')
        sheet = source_file.sheet_by_index(0)
        for i in range(1, sheet.nrows):
            content=sheet.cell(i,1).value
            if content.find('、')==-1 and content.find(';')==-1 and content.find('杀人')!=-1:
                for j in range(1,sheet.ncols):
                    target_sheet.write(target_col,j-1,sheet.cell(i,j).value)
                target_col+=1
    target_file.save(source_foldname+'/all_data.xls')

        
def main():
    case_name=input('please input case name:')
    data_type=int(input('please input data_type:'))
    target_case=DealRawData(case_name)
    target_case.xls2txt(data_type=data_type)

if __name__=='__main__':
    main()