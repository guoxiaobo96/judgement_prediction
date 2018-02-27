"""This is used to do some preparations for the training material
including changing from xls to txt,participating, labeling and so on
the package include xlrd to change from xls to txt,
jieba to participate the txt"""

import xlrd
import jieba
import os
import sys
import random

def xls2txt(foldname):
    """this part is used to change from xls to txt"""
    if foldname!='':
        source_foldname = foldname
    else:
        source_foldname = "D:/案件数据/故意杀人案"
    source_file_list = os.listdir(source_foldname)
    target_filename = source_foldname+"/conclusion.txt"
    case_number = 0
    feature_result_matrix=str()
    feature_list=['未遂','重伤','谅解','自首','恶劣','累犯','残忍','赔偿','过错']
    for source_filename in source_file_list:
        source_filename = source_foldname+"/"+source_filename
        source_file = xlrd.open_workbook(filename=source_filename,encoding_override='utf-8')
        sheet = source_file.sheet_by_index(0)
        content = str()
        for _ in range(1, sheet.nrows):
            if sheet.cell(_, 1).value.find('、')!=-1:
                continue
            case_number +=1
            temp_feature=str()
            content = sheet.cell(_, 13).value
            for i, feature in enumerate(feature_list):
                if content.find(feature)!=-1:
                    temp_feature+='1,'
                else:
                    temp_feature+='0,'
            temp_feature+=getLabel(sheet.cell(_, 14).value)
            feature_result_matrix+=temp_feature
    with open(file=target_filename, mode="a",encoding='utf-8') as target_file:
        target_file.write(feature_result_matrix)

    return "xls2txt finish"

def getLabel(content):
    """this part is used to label the case and it should be changed when dealing with different cases"""
    if content.find("死刑")==-1:
        if content.find("缓刑")!=-1:
            return '0\n'
        elif content.find("无期徒刑")!=-1:
            return '3\n'
        elif content.find("十")!=-1:
            return '2\n'
        else:
            return '1\n'
    else:
        return '4\n'

def main():
    """main"""
    print(xls2txt("D:/案件数据/故意杀人案"))
#    print(labelCase("D:/案件数据/故意杀人案"))

if __name__ == '__main__':
    main()
