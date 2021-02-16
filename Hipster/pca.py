#coding=utf-8
import requests
import time
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import networkx as nx
import argparse
import csv
import itertools
import os

from sklearn.decomposition import SparsePCA
from sklearn.cluster import Birch
from sklearn import preprocessing
from numpy import mean

def Z_Score(data):
    lenth = len(data)
    total = sum(data)
    ave = float(total)/lenth
    tempsum = sum([pow(data[i] - ave,2) for i in range(lenth)])
    tempsum = pow(float(tempsum)/lenth,0.5)
    for i in range(lenth):
        data[i] = (data[i] - ave)/tempsum
    return data

def attributed_graph(faults_name):
    # build the attributed graph 
    # input: prefix of the file
    # output: attributed graph

    filename = faults_name + '_mpg.csv'
    df = pd.read_csv(filename)

    DG = nx.DiGraph()    
    for index, row in df.iterrows():
        source = row['source']
        destination = row['destination']
        if 'rabbitmq' not in source and 'rabbitmq' not in destination and 'db' not in destination and 'db' not in source:
            DG.add_edge(source, destination)

    for node in DG.nodes():
        if 'kubernetes' in node: 
            DG.nodes[node]['type'] = 'host'
        else:
            DG.nodes[node]['type'] = 'service'
                
    return DG 

def sigmoid(x):
    return np.around(1./(1+np.exp(-x)))

# latency = [[200,200,180,117], [16,20,25,20], [17,30,22,24], [150,180,160,175],[48,20,17,32]]

def PCA():
    latency = pd.read_csv('latency.csv').T
    l1 = latency.iloc[:, :1]
    # new1 = np.array(l1)
    index = latency.index.values
    latency = latency.iloc[:, 1:].fillna(0)
    X = np.array(latency)
    pca = SparsePCA(n_components=1)
    pca.fit(X)    
    newX=pca.fit_transform(X)

    max_abs_scaler = preprocessing.MinMaxScaler()
    newX = max_abs_scaler.fit_transform(newX)
    pca_df = pd.DataFrame(newX)
    df1= pca_df.set_index(index)
    print('\n', df1)
    return df1

def birch_ad_with_cluster_nums():    
    cluster_nums = {}
    latency_df = pd.read_csv('latency.csv')
    for svc, latency in latency_df.iteritems():
        # No anomaly detection in db
        if svc != 'timestamp' and 'Unnamed' not in svc and 'rabbitmq' not in svc and 'db' not in svc:
            latency = latency.rolling(window=12, min_periods=1).mean()
            x = np.array(latency)
            x = np.where(np.isnan(x), 0, x)

            normalized_x = preprocessing.normalize([x])

            X = normalized_x.reshape(-1,1)
            brc = Birch(branching_factor=50, n_clusters=None, threshold=0.05, compute_labels=True)
            brc.fit(X)
            brc.predict(X)

            labels = brc.labels_
            # print(labels)
          # centroids = brc.subcluster_centers_
            n_clusters = np.unique(labels).size
            cluster_nums[svc] = n_clusters
    print('\n', cluster_nums)
    return cluster_nums

def calc_score(faults_name, cluster_nums):

    fault = faults_name.replace('./data/', '')
    foo = {}

    DG = attributed_graph(faults_name)
    # a = list(nx.edge_dfs(DG,'frontend'))
    # print('\n DG:', a)
    df_pca = PCA()
    index = df_pca.index.values
    
    # MD!!!!! 我发现我代码里出现致命错误，我一开始先入为主认为 anomaly 就是 fault 然后计算累加结果，那肯定是 anomaly 的分数最高啊
    p_scores = {}
    for node in ['adservice', 'shippingservice', 'cartservice', 'paymentservice', 'recommendationservice','emailservice', 'redis-cart', 'checkoutservice', 'currencyservice', 'frontend']:
    # for node in ['adservice', 'shippingservice']:
        foo[node] = 0

        for i in index:
            if 'Unnamed' not in i:
                p_score = 1 - (df_pca.loc[i,0])
                endpoint = str(i).split('_')[1]
                if (node == endpoint):
                    foo[node] = foo[node] + (p_score * cluster_nums[i])

        # for path in nx.all_simple_paths(DG, source='frontend', target=node):
        #     print(path)
        #     for i in list(itertools.combinations(path, 2)):
        #         trace = i[0] + '_' + i[1]
        #         if trace in index and node not in index:
        #             
        #             print(trace, p_score)
        #             # print(p_score)
        #             foo[node] = foo[node] + (p_score * cluster_nums[trace])               

        # 这里要用出度
        degree = DG.out_degree[node] + 1
        foo[node] = foo[node] / degree

    print('\n', foo)
    return foo

cluster_nums = birch_ad_with_cluster_nums()
calc_score('./data/adservice', cluster_nums)