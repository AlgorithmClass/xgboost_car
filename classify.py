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
    'objective':'reg:linear'
}
num_round=2
watchlist=[(dvalid,'eval'),(dtrain,'train')]
bst=xgb.train(param,dtrain,num_round,watchlist)
preds=bst.predict(dvalid)
