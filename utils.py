# -*- coding: utf-8 -*-

import xgboost as xgb
import numpy as np


def testPred(dres, filename):
    dvalid = open(filename, "r")

    realRes = []
    for line in dvalid:
        realRes.append(int(line.split(" ")[0]))

    l = len(realRes)
    rmse_sum = 0.0
    for i in range(l):
        rmse_sum += (realRes[i] - dres[i]) * (realRes[i] - dres[i])

    print (rmse_sum / l) ** 0.5


if __name__ == "__main__":
    testPred(1, "valid.txt")
