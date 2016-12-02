'''
python3
生成xgboost格式数据
'''
import pandas as pd

#训练集
def get_train_data():
    train_csv=pd.read_csv("train.csv")
    train_csv=train_csv.drop('Id',axis=1)
    columns=[]
    for col in train_csv:
        if str(train_csv[col][0])[0].isalpha():
            t=pd.get_dummies(train_csv[col])
            columns.append(t)
        else: columns.append(train_csv[col])
    train_cat=pd.concat(columns,axis=1)
    return train_cat

















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





