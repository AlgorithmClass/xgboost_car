import mapfeat
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.cross_validation import KFold
#from sklearn.linear_model import LinearRegression#####################
#from sklearn import cross_validation
#from sklearn.datasets import load_svmlight_file


#libsvm_data=pd.read_csv("libsvm_data.csv",sep=' ',header=None)
train_data=mapfeat.get_train_data()
y_data=np.array(train_data['Score'],dtype=int)
X_data=np.array(train_data.drop('Score',axis=1),dtype=int)
#load_svmlight_file()

'''
###############
clf=LinearRegression()
score=cross_validation.cross_val_score(clf,X_data,y_data,scoring='mean_squared_error')
score=-score
print(score)
print(np.sqrt(score))
input()
'''

num_round=700
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
pow_num_round=172
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
    l=len(real)
    rmse_sum=0.0
    for i in range(l):
        rmse_sum+=np.square(pred[i]-real[i])
    return np.sqrt(rmse_sum/l)

def train_pred(y_data,X_data,param,num_round):
    k_fold=KFold(data_length,num_fold,random_state=20)
    pred=[]
    real=[]
    for train,valid in k_fold:
        X_train=X_data[train]
        y_train=y_data[train]
        X_valid=X_data[valid]
        y_valid=y_data[valid]

        dtrain=xgb.DMatrix(X_train,label=y_train)
        dvalid=xgb.DMatrix(X_valid,label=y_valid)
        watchlist=[(dvalid,'eval'),(dtrain,'train')]
        bst=xgb.train(param,dtrain,num_round,watchlist,learning_rates=None)
        valid_pred=bst.predict(dvalid)
        valid_real=y_valid
        pred.append(valid_pred)
        real.append(valid_real)
        input('跑完一圈！回车继续...')
    return pred,real
        #rmse_score.append(rmse(valid_pred,valid_real))


#原始数据：
rmse_cross=[]
pred,real=train_pred(y_data,X_data,param,num_round)
for i in range(len(pred)):
    rmse_cross.append(rmse(pred[i],real[i]))
print('原始数据：',rmse_cross)
rmse_cross=np.array(rmse_cross)
print('\n平均rmse:',rmse_cross.sum()/num_fold)
input()

#poi
rmse_cross=[]
pred,real=train_pred(y_data,X_data,poi_param,poi_num_round)
for i in range(len(pred)):
    rmse_cross.append(rmse(pred[i],real[i]))
print('\npoi：',rmse_cross)
rmse_cross=np.array(rmse_cross)
print('\n平均rmse:',rmse_cross.sum()/num_fold)
input()

#log
rmse_cross=[]
y_log=np.log1p(y_data)
pred,real=train_pred(y_log,X_data,log_param,log_num_round)
for i in range(len(pred)):
    pred[i]=np.expm1(pred[i])
    real[i]=np.expm1(real[i])
for i in range(len(pred)):
    rmse_cross.append(rmse(pred[i],real[i]))
print('\nlog：',rmse_cross)
rmse_cross=np.array(rmse_cross)
print('\n平均rmse:',rmse_cross.sum()/num_fold)
input()

#sqrt 
rmse_cross=[]
y_sqrt=np.sqrt(y_data)
pred,real=train_pred(y_sqrt,X_data,sqrt_param,sqrt_num_round)
for i in range(len(pred)):
    pred[i]=np.square(pred[i])
    real[i]=np.square(real[i])
for i in range(len(pred)):
    rmse_cross.append(rmse(pred[i],real[i]))
print('\nsqrt：',rmse_cross)
rmse_cross=np.array(rmse_cross)
print('\n平均rmse:',rmse_cross.sum()/num_fold)
input()

#pow 
rmse_cross=[]
y_squre=np.square(y_data)
pred,real=train_pred(y_squre,X_data,pow_param,pow_num_round)
for i in range(len(pred)):
    pred[i]=np.sqrt(pred[i])
    real[i]=np.sqrt(real[i])
for i in range(len(pred)):
    rmse_cross.append(rmse(pred[i],real[i]))
print('\npow：',rmse_cross)
rmse_cross=np.array(rmse_cross)
print('\n平均rmse:',rmse_cross.sum()/num_fold)
input()
