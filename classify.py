import xgboost as xgb

dtrain=xgb.DMatrix("train.txt")
dvalid=xgb.DMatrix("valid.txt")
param={
    'max_depth':3,
    'eta':1.0,
    'gamma':1.0,
    'min_child_weight':1,
    'save_period':0,
    'booster':'gbtree',
    'objective':'reg:linear',
    #'objective':'count:poisson'
}
num_round=10
learning_rates=[1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]
watchlist=[(dvalid,'eval'),(dtrain,'train')]
bst=xgb.train(param,dtrain,num_round,watchlist,learning_rates=learning_rates)
preds=bst.predict(dvalid)
