from sklearn.svm import SVC
import xlrd
source_foldname = "D:/案件数据/故意杀人案"
train_content_filename = source_foldname+"/train.xls"
test_content_filename = source_foldname+"/test.xls"

x_train_temp = list()
y_train_temp = list()
x_test_temp = list()
y_test_temp = list()

train_file = xlrd.open_workbook(filename=train_content_filename)
sheet = train_file.sheet_by_index(0)
for _ in range(1, sheet.nrows):
    temp_feature=list()
    for i in range(9):
        temp_feature.append(sheet.cell(_, i).value)
    y_train_temp.append(sheet.cell(_, 9).value)
    x_train_temp.append(temp_feature)

test_file = xlrd.open_workbook(filename=test_content_filename)
sheet = test_file.sheet_by_index(0)
for _ in range(1, sheet.nrows):
    temp_feature=list()
    for i in range(9):
        temp_feature.append(sheet.cell(_, i).value)
    y_test_temp.append(sheet.cell(_, 9).value)
    x_test_temp.append(temp_feature)

svclf = SVC(kernel = 'linear') 
svclf.fit(x_train_temp,y_train_temp)
preds = svclf.predict(x_test_temp)
num = 0
preds = preds.tolist()
for i,pred in enumerate(preds):
    if int(pred) == int(y_test_temp[i]):
        num += 1
print ('precision_score: '+str(float(num) / len(preds)))