import xgboost as xgb
import utils


def classify(maxdepth=3):
    dtrain = xgb.DMatrix("train.txt")
    dvalid = xgb.DMatrix("valid.txt")
    param = {
        'max_depth': maxdepth,
        'eta': 1,
        'gamma': 1,
        'min_child_weight': 1,
        'save_period': 0,
        'booster': 'gbtree',
        'objective': 'reg:linear',
    }

    num_round = 15
    learning_rates = [1.5, 1.4, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1]
    watchlist = [(dvalid, 'eval'), (dtrain, 'train')]
    bst_o = xgb.train(param, dtrain, num_round, watchlist, learning_rates=learning_rates)

    param = {
        'max_depth': maxdepth,
        'eta': 1,
        'gamma': 1,
        'min_child_weight': 1,
        'save_period': 0,
        'booster': 'gbtree',
         'objective':'count:poisson'
    }

    num_round = 15
    learning_rates = [1.5, 1.4, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    watchlist = [(dvalid, 'eval'), (dtrain, 'train')]
    bst_p = xgb.train(param, dtrain, num_round, watchlist, learning_rates=learning_rates)

    preds_vo = bst_o.predict(dvalid)
    preds_vp = bst_p.predict(dvalid)
    utils.testPred(preds_vo, "valid.txt")
    utils.testPred(preds_vp, "valid.txt")
    utils.testPred((preds_vo + preds_vp) / 2, "valid.txt")

    preds_to = bst_o.predict(dtrain)
    preds_tp = bst_p.predict(dtrain)
    utils.testPred(preds_to, "train.txt")
    utils.testPred(preds_tp, "train.txt")
    utils.testPred((preds_to + preds_tp) / 2, "train.txt")



if __name__ == "__main__":
    classify()
