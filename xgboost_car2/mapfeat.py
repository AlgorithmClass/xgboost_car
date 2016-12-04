'''
python3
生成xgboost格式数据
'''
import pandas as pd

def xg_data(data_csv):
    '''
    用于生成xgboost可使用的数据格式
    '''
    columns=[]
    for col in data_csv:
        if str(data_csv[col][0])[0].isalpha():
            t=pd.get_dummies(data_csv[col])
            columns.append(t)
        else: columns.append(data_csv[col])
    return pd.concat(columns,axis=1) #将列拼接在一起


def get_data():
    '''
    对训练集和测试集进行处理并返回所需格式
    '''
    train_csv=pd.read_csv("train.csv")
    test_csv=pd.read_csv("test.csv")
    train_csv=train_csv.drop('Id',axis=1)
    train_cat=xg_data(train_csv)
    test_cat=xg_data(test_csv)
    return train_cat,test_cat

















'''




order_alpha=[]
order_num=[]
alpha_one_hot={}
row=[]

train_csv=open("train.csv")

i=0
for line in train_csv:
    if i==0:
        i+=1
        continue
    else:
        row=line.split(',')
        break
#print(row)
features=len(row)
for i in range(2,features):
    if(row[i][0].isalpha()):
        order_alpha.append(i)
    else:order_num.append(i)

assert((len(order_alpha)+len(order_num))==(features-2))

#print(order_alpha)

for i in order_alpha:
        alpha_one_hot[i]={}

train_csv.close()
train_csv=open("train.csv")
for line in train_csv:
    line=line.split(',')
    if line[0][0]=='I': continue
    for i in order_alpha:
        c=line[i][0]
        if (c not in alpha_one_hot[i]):
            alpha_one_hot[i][c]=len(alpha_one_hot[i])
       
alpha_begin=len(order_num)+1
for i in order_alpha:
    for v, k in sorted( alpha_one_hot[i].items(), key = lambda x:x[1] ):
        alpha_one_hot[i][v]=k+alpha_begin
    alpha_begin+=len(alpha_one_hot[i])


'''
'''
for i in order_alpha:
    print(alpha_one_hot[i])
input()
'''
'''
libsvm_data=open("libsvm_data.csv","w")
train_csv.close()
train_csv=open("train.csv")
for line in train_csv:
    line=line.split(',')
    if line[0][0]=='I': continue
    libsvm_data.write(line[1])
    for i,k in enumerate (order_num):
        libsvm_data.write(" %d:%s"%((i+1),line[k]))
    for j in order_alpha:
        libsvm_data.write(" %d:1"%alpha_one_hot[j][line[j][0]])
    libsvm_data.write("\n")
train_csv.close()
libsvm_data.close()


'''





