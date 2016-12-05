import mapfeat
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.cross_validation import KFold
#from sklearn.datasets import dump_svmlight_file
#from sklearn.linear_model import LinearRegression#####################
#from sklearn import cross_validation
#from sklearn.datasets import load_svmlight_file


#libsvm_data=pd.read_csv("libsvm_data.csv",sep=' ',header=None)

#准备数据
train_data,test_data=mapfeat.get_data()
y_data=np.array(train_data['Score'],dtype=int)
X_data=np.array(train_data.drop('Score',axis=1),dtype=int)
X_test=np.array(test_data,dtype=int)
id_test=X_test[:,0]
X_test=X_test[:,1:]
#load_svmlight_file()

'''
###############
clf=LinearRegression()
score=cross_validation.cross_val_score(clf,X_data,y_data,scoring='mean_squared_error')
score=-score
print(score)
print(np.sqrt(score))
#input()
'''

num_round=600
param={
        'max_depth':2,
        'eta':0.1,
        'gamma':0,
        'min_child_weight':0,
        'save_period':0,
        'booster':'gbtree',
        'silent':1,
        "seed": 0,
        #'subsample':0.7,
        #'colsample_bytree':0.9,
        #'colsample_bylevel':0.9,
        'lambda':6.0,
        'alpha':4.0,
        'objective':'reg:linear',
        #'objective':'count:poisson'
    }

log_num_round=81
log_param={
    'max_depth':5,
    'eta':0.08,
    'gamma':0,
    'min_child_weight':0,
    'save_period':0,
    'booster':'gbtree',
    'silent':1,
    "seed": 0,
    #'subsample':0.7,
    #'colsample_bytree':0.9,
    #'colsample_bylevel':0.9,
    'lambda':7.0,
    'alpha':7.0,
    'objective':'reg:linear',
    #'objective':'count:poisson'
}
sqrt_num_round=321
sqrt_param={
    'max_depth':3,
    'eta':0.08,
    'gamma':0,
    'min_child_weight':1,
    'save_period':0,
    'booster':'gbtree',
    'silent':1,
    "seed": 0,
    #'subsample':0.7,
    #'colsample_bytree':0.9,
    #'colsample_bylevel':0.9,
    'lambda':7.0,
    'alpha':3.0,
    'objective':'reg:linear',
    #'objective':'count:poisson'
}
pow_num_round=20 #172
pow_param={
    'max_depth':3,
    'eta':0.08,
    'gamma':0,
    'min_child_weight':1,
    'save_period':0,
    'booster':'gbtree',
    'silent':1,
    "seed": 0,
    #'subsample':0.7,
    #'colsample_bytree':0.9,
    #'colsample_bylevel':0.9,
    'lambda':5.0,
    #'alpha':1.0,
    'objective':'reg:linear',
    #'objective':'count:poisson'
}
poi_num_round=607
poi_param={
    'max_depth':2,
    'eta':0.2,
    'gamma':0,
    'min_child_weight':1,
    'save_period':0,
    'booster':'gbtree',
    'silent':1,
    "seed": 0,
    #'subsample':0.7,
    #'colsample_bytree':0.9,
    #'colsample_bylevel':0.9,
    'lambda':2.8,
    #'alpha':3.0,
    #'objective':'reg:linear',
    'objective':'count:poisson'
}


data_length=len(y_data)
num_fold=3

def rmse(pred,real):
    '''
    rmse计算函数
    '''
    #pred=np.round(pred)
    l=len(real)
    rmse_sum=0.0
    for i in range(l):
        rmse_sum+=np.square(pred[i]-real[i])
    return np.sqrt(rmse_sum/l)

def train_pred(X_test,y_data,X_data,param,num_round):
    '''
    通过交叉验证进行调参，得到效果较好的模型
    返回值：训练数据的预测值，真实值，测试数据的预测值
    '''
    k_fold=KFold(data_length,num_fold,random_state=20)
    pred=[]
    real=[]
    model=[]
    dtest=xgb.DMatrix(X_test)
    test_score=[]
    for train,valid in k_fold:
        X_train=X_data[train]
        y_train=y_data[train]
        X_valid=X_data[valid]
        y_valid=y_data[valid]

        dtrain=xgb.DMatrix(X_train,label=y_train)
        dvalid=xgb.DMatrix(X_valid,label=y_valid)
        watchlist=[(dvalid,'eval'),(dtrain,'train')]
        bst=xgb.train(param,dtrain,num_round,watchlist,learning_rates=None)
        test_score.append(bst.predict(dtest))
        valid_pred=bst.predict(dvalid)
        valid_real=y_valid
        pred.append(valid_pred)
        real.append(valid_real)
        #input('跑完一圈！回车继续...')
    
    return pred,real,test_score
        #rmse_score.append(rmse(valid_pred,valid_real))  np.array(test_score).sum(axis=0)/num_fold


#test_results=[]

#原始数据：
rmse_cross=[]
pred,real,test_result=train_pred(X_test,y_data,X_data,param,num_round)
test_results=test_result
for i in range(len(pred)):
    rmse_cross.append(rmse(pred[i],real[i]))
print('原始数据：',rmse_cross)
rmse_cross=np.array(rmse_cross)
print('\n平均rmse:',rmse_cross.sum()/num_fold)
#input('回车继续...')

#泊松回归
rmse_cross=[]
pred,real,poi_test_result=train_pred(X_test,y_data,X_data,poi_param,poi_num_round)
test_results.extend(poi_test_result)
for i in range(len(pred)):
    rmse_cross.append(rmse(pred[i],real[i]))
print('\npoi：',rmse_cross)
rmse_cross=np.array(rmse_cross)
print('\n平均rmse:',rmse_cross.sum()/num_fold)
#input('回车继续...')

#对score取对数后再训练
rmse_cross=[]
y_log=np.log1p(y_data)
pred,real,_=train_pred(X_test,y_log,X_data,log_param,log_num_round)
for i in range(len(pred)):
    pred[i]=np.expm1(pred[i])
    real[i]=np.expm1(real[i])
for i in range(len(pred)):
    rmse_cross.append(rmse(pred[i],real[i]))
print('\nlog：',rmse_cross)
rmse_cross=np.array(rmse_cross)
print('\n平均rmse:',rmse_cross.sum()/num_fold)
#input('回车继续...')

#对score开方后再训练
rmse_cross=[]
y_sqrt=np.sqrt(y_data)
pred,real,_=train_pred(X_test,y_sqrt,X_data,sqrt_param,sqrt_num_round)
for i in range(len(pred)):
    pred[i]=np.square(pred[i])
    real[i]=np.square(real[i])
for i in range(len(pred)):
    rmse_cross.append(rmse(pred[i],real[i]))
print('\nsqrt：',rmse_cross)
rmse_cross=np.array(rmse_cross)
print('\n平均rmse:',rmse_cross.sum()/num_fold)
#input('回车继续...')

#对score平方后再训练
rmse_cross=[]
y_squre=np.square(y_data+4.0)
pred,real,_=train_pred(X_test,y_squre,X_data,pow_param,pow_num_round)
for i in range(len(pred)):
    pred[i]=np.sqrt(pred[i])-4.0
    real[i]=np.sqrt(real[i])-4.0
for i in range(len(pred)):
    rmse_cross.append(rmse(pred[i],real[i]))
print('\npow：',rmse_cross)
rmse_cross=np.array(rmse_cross)
print('\n平均rmse:',rmse_cross.sum()/num_fold)
#input('回车继续...')

#预测结果：
test_results=np.array(test_results).sum(axis=0)/len(test_results)
#print(test_results)
#test_results=test_results.round().astype(int)
#print(test_results)
#input()
Test_Results=pd.DataFrame({"Id":id_test,"Score":test_results})
Test_Results.to_csv("1501214415_predict.csv",sep=',',index=False)
print("结束，结果保存在1501214415_predict.csv")

'''
train_csv=pd.read_csv("train.csv")
real=np.array(train_csv["Score"])
print(rmse(_test_results,real))
'''


