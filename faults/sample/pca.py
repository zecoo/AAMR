#coding=utf-8
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.decomposition import SparsePCA


def normalization(data):
    _range = np.max(data) - np.min(data)
    return 1-((data - np.min(data)) / _range)

def Z_Score(data):
    lenth = len(data)
    total = sum(data)
    ave = float(total)/lenth
    tempsum = sum([pow(data[i] - ave,2) for i in range(lenth)])
    tempsum = pow(float(tempsum)/lenth,0.5)
    for i in range(lenth):
        data[i] = (data[i] - ave)/tempsum
    return data


def sigmoid(x):
    return np.around(1./(1+np.exp(-x)))

# latency = [[200,200,180,117], [16,20,25,20], [17,30,22,24], [150,180,160,175],[48,20,17,32]]
# 实验发现PCA对值比较大的数据很敏感，很容易就搞成PC了。

latency = pd.read_csv('pca_test.csv')
# 数据转置一下
T_latency = latency.T


X = np.array(T_latency)  #导入数据，维度为4
pca = SparsePCA(n_components=1)   #降到2维
pca.fit(X)                  #训练
newX=pca.fit_transform(X)   #降维后的数据
# PCA(copy=True, n_components=2, whiten=False)
# print(pca.explained_variance_ratio_)  #输出贡献率
print(sigmoid(newX))                  #输出降维后的数据