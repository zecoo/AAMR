#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: li
"""

import pandas as pd
import numpy as np
import networkx as nx
import csv
# import matplotlib.pyplot as plt

from sklearn.cluster import Birch
from sklearn import preprocessing


smoothing_window = 12

# 返回的 anomalies 直接就把可能的错误列出来了：['front-end_carts', 'front-end_orders', 'orders_carts', 'orders_payment', 'orders_shipping']

# Anomaly Detection
def birch_ad_with_smoothing(latency_df, threshold):
    # anomaly detection on response time of service invocation. 
    # input: response times of service invocations, threshold for birch clustering
    # output: anomalous service invocation
    
    anomalies = []
    count = 1
    for svc, latency in latency_df.iteritems():
        # No anomaly detection in db
        if svc != 'timestamp' and 'Unnamed' not in svc and 'rabbitmq' not in svc and 'db' not in svc:
            count = count + 1

            # 这段代码对 latency 的影响好像几乎可以不计
            # 
            # latency 片段是这样：（取前3行，共30行）
            # 
            # Name: front-end_user, dtype: float64
            # 0      12.101140
            # 1       7.617647
            # 2      11.812500
            # 3      17.202381
            # 
            latency = latency.rolling(window=smoothing_window, min_periods=1).mean()
            
            # x 就是把 latency 中 Name、dtype 等信息抹掉，并转为一个 array 形式的数据
            x = np.array(latency)
            x = np.where(np.isnan(x), 0, x)

            # normalize 就是归一化，不然都是100，1k的话计算起来太麻烦
            normalized_x = preprocessing.normalize([x])

            # reshape 这个操作可以百度一下，看看就清楚了
            # 简单来说就是 把[x, y ,z] 这样的数据转为：
            # [x,
            #  y,
            #  z]

            X = normalized_x.reshape(-1,1)

#            threshold = 0.05

            # 所以聚类的对象是什么？应该不是每一行，而是整体
            # 聚类的结果分为几类，如果某一行都是一种类型，表示没有异常，如果有多种类型，说明有 anomaly
            brc = Birch(branching_factor=50, n_clusters=None, threshold=threshold, compute_labels=True)
            brc.fit(X)
            brc.predict(X)

            labels = brc.labels_
            
#            centroids = brc.subcluster_centers_

            # np.unique 的作用是去除数组中重复的元素

            n_clusters = np.unique(labels).size
            # print(n_clusters)
            if n_clusters > 1:
                anomalies.append(svc)
    return anomalies

# 读取 source 和 destination 的 scv file
# 最后把 source 的 df 拼接到 destination 的后面
# 并不是拼接到后面，而是 source + destination 就是某个服务的相应时间
# 实际看了一下数据我才知道

def rt_invocations(faults_name):
    # retrieve the response time of each invocation from data collection
    # input: prefix of the csv file
    # output: round-trip time
    
    latency_filename = faults_name + '_latency_source_50.csv'  # inbound
    latency_df_source = pd.read_csv(latency_filename) 
    # print('source')
    # print(latency_df_source)
    latency_df_source['unknown_front-end'] = 0
    
    latency_filename = faults_name + '_latency_destination_50.csv' # outbound
    latency_df_destination = pd.read_csv(latency_filename) 
    # print('destination')
    # print(latency_df_destination)
    latency_df = latency_df_destination.add(latency_df_source)    
    latency_df.to_csv('%s_latency.csv'%faults_name, index=None)
    # print('result')
    # print(latency_df)
    return latency_df

# 这里是通过 mpg.scv 构建有向图，用的工具是networkx工具包，
# 那么我有一个问题，就是 mpg.svc 是怎么得到的？

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
            
    # plt.figure(figsize=(9,9))
    # nx.draw(DG, with_labels=True, font_weight='bold')
    # pos = nx.spring_layout(DG)
    # nx.draw(DG, pos, with_labels=True, cmap = plt.get_cmap('jet'), node_size=1500, arrows=True, )
    # labels = nx.get_edge_attributes(DG,'weight')
    # nx.draw_networkx_edge_labels(DG,pos,edge_labels=labels)
    # plt.show()
                
    return DG 

def node_weight(svc, anomaly_graph, baseline_df, faults_name):

    #Get the average weight of the in_edges
    in_edges_weight_avg = 0.0
    num = 0
    for u, v, data in anomaly_graph.in_edges(svc, data=True):
#        print(u, v)
        num = num + 1
        in_edges_weight_avg = in_edges_weight_avg + data['weight']
    if num > 0:
        in_edges_weight_avg  = in_edges_weight_avg / num

    filename = faults_name + '_' + svc + '.csv'
    df = pd.read_csv(filename)
    node_cols = ['node_cpu', 'node_network', 'node_memory']
    max_corr = 0.01
    metric = 'node_cpu'
    for col in node_cols:
        temp = abs(baseline_df[svc].corr(df[col]))
        if temp > max_corr:
            max_corr = temp
            metric = col
    data = in_edges_weight_avg * max_corr
    return data, metric

def svc_personalization(svc, anomaly_graph, baseline_df, faults_name):

    filename = faults_name + '_' + svc + '.csv'
    df = pd.read_csv(filename)
    ctn_cols = ['ctn_cpu', 'ctn_network', 'ctn_memory']
    max_corr = 0.01
    metric = 'ctn_cpu'
    for col in ctn_cols:
        temp = abs(baseline_df[svc].corr(df[col]))     
        if temp > max_corr:
            max_corr = temp
            metric = col


    edges_weight_avg = 0.0
    num = 0
    for u, v, data in anomaly_graph.in_edges(svc, data=True):
        num = num + 1
        edges_weight_avg = edges_weight_avg + data['weight']

    for u, v, data in anomaly_graph.out_edges(svc, data=True):
        if anomaly_graph.nodes[v]['type'] == 'service':
            num = num + 1
            edges_weight_avg = edges_weight_avg + data['weight']

    edges_weight_avg  = edges_weight_avg / num

    personalization = edges_weight_avg * max_corr

    return personalization, metric

def calc_score(faults_name):
    
    fault = faults_name.replace('./MicroRCA_Online/','')

    latency_filename = faults_name + '_latency_source_50.csv'  # inbound
    latency_df_source = pd.read_csv(latency_filename)

    latency_filename = faults_name + '_latency_destination_50.csv' # outbound
    latency_df_destination = pd.read_csv(latency_filename) 

    latency_df_source.loc['all'] = latency_df_source.apply(lambda x:x.sum())

    latency_df_destination.loc['all'] = latency_df_destination.apply(lambda x:x.sum())

    df_data = pd.DataFrame(columns=['svc', 'ratio'])
    df_data = latency_df_source.loc['all'] / latency_df_destination.loc['all']

    ratio = df_data.to_dict()
    scores = {}

    for key in list(ratio.keys()):
        if 'db' in key or 'rabbitmq' in key or 'Unnamed' in key:
            del ratio[key]
        else:
            svc_name = key.split('_')[1]
            ratio.update({svc_name: ratio.pop(key)})
    
    print('\nratio: ', ratio)

    DG = attributed_graph(faults_name)

    print('\ndegree: ', DG.degree)
    up = pd.DataFrame(ratio, index=[0]).T
    down  = pd.DataFrame(dict(DG.degree), index=[0]).T
    score = (up / down).dropna().to_dict()
    score = score[0]

    print('\nscore:', score)

    score_list = []
    for svc in score:
        item = (svc, score[svc])
        score_list.append(score[svc])

    score_arr = np.array(score_list)

    print(score_arr)

    z_score = []
    for x in score_arr:
        x = float(x - score_arr.mean())/score_arr.std() + 0.3
        z_score.append(x)
    
    print('\nz_score: ', z_score)

    n = 0
    for svc in score:
        score.update({svc: z_score[n]})
        n = n + 1

    print('\nnew score: ',score)

    return score

def anomaly_subgraph(DG, anomalies, latency_df, faults_name, alpha):
    # Get the anomalous subgraph and rank the anomalous services
    # input: 
    #   DG: attributed graph
    #   anomlies: anoamlous service invocations
    #   latency_df: service invocations from data collection
    #   agg_latency_dff: aggregated service invocation
    #   faults_name: prefix of csv file
    #   alpha: weight of the anomalous edge
    # output:
    #   anomalous scores 
    
    # Get reported anomalous nodes
    edges = []
    nodes = []
#    print(DG.nodes())
    baseline_df = pd.DataFrame()
    edge_df = {}
    for anomaly in anomalies:
        edge = anomaly.split('_')
        edges.append(tuple(edge))
#        nodes.append(edge[0])
        svc = edge[1]
        nodes.append(svc)
        baseline_df[svc] = latency_df[anomaly]
        edge_df[svc] = anomaly

#    print('edge df:', edge_df)
    nodes = set(nodes)
#    print(nodes)

    personalization = {}
    for node in DG.nodes():
        if node in nodes:
            personalization[node] = 0

    # Get the subgraph of anomaly
    anomaly_graph = nx.DiGraph()
    for node in nodes:
#        print(node)
        for u, v, data in DG.in_edges(node, data=True):
            edge = (u,v)
#            print(edge)
            if edge in edges:
                data = alpha
            else:
                normal_edge = u + '_' + v
                data = baseline_df[v].corr(latency_df[normal_edge])

            data = round(data, 3)
            anomaly_graph.add_edge(u,v, weight=data)
            anomaly_graph.nodes[u]['type'] = DG.nodes[u]['type']
            anomaly_graph.nodes[v]['type'] = DG.nodes[v]['type']

       # Set personalization with container resource usage
        for u, v, data in DG.out_edges(node, data=True):
            edge = (u,v)
            if edge in edges:
                data = alpha
            else:

                if DG.nodes[v]['type'] == 'host':
                    data, col = node_weight(u, anomaly_graph, baseline_df, faults_name)
                else:
                    normal_edge = u + '_' + v
                    data = baseline_df[u].corr(latency_df[normal_edge])
            data = round(data, 3)
            anomaly_graph.add_edge(u,v, weight=data)
            anomaly_graph.nodes[u]['type'] = DG.nodes[u]['type']
            anomaly_graph.nodes[v]['type'] = DG.nodes[v]['type']


    for node in nodes:
        max_corr, col = svc_personalization(node, anomaly_graph, baseline_df, faults_name)
        personalization[node] = max_corr / anomaly_graph.degree(node)
#        print(node, personalization[node])

    anomaly_graph = anomaly_graph.reverse(copy=True)

    edges = list(anomaly_graph.edges(data=True))

    for u, v, d in edges:
        if anomaly_graph.nodes[node]['type'] == 'host':
            anomaly_graph.remove_edge(u,v)
            anomaly_graph.add_edge(v,u,weight=d['weight'])

#    plt.figure(figsize=(9,9))
##    nx.draw(DG, with_labels=True, font_weight='bold')
#    pos = nx.spring_layout(anomaly_graph)
#    nx.draw(anomaly_graph, pos, with_labels=True, cmap = plt.get_cmap('jet'), node_size=1500, arrows=True, )
#    labels = nx.get_edge_attributes(anomaly_graph,'weight')
#    nx.draw_networkx_edge_labels(anomaly_graph,pos,edge_labels=labels)
#    plt.show()
#
##    personalization['shipping'] = 2
#    print('Personalization:', personalization)

    # personalized pagerank 体现在这里
    # 那么重点中的重点就是这个 anomaly_graph 另外这个 nx 工具包里的 PPR 的输入输出分别是什么

    print('\nanomaly graph: ', anomaly_graph.adj)

    personalization = calc_score(faults_name)
    print('\npersonalization: ', personalization)
    
    anomaly_score = nx.pagerank(anomaly_graph, alpha=0.85, personalization=personalization, max_iter=1000)

    anomaly_score = sorted(anomaly_score.items(), key=lambda x: x[1], reverse=True)

#    return anomaly_graph
    return anomaly_score

def print_rank(anomaly_score, target):
    num = 10
    for idx, anomaly_target in enumerate(anomaly_score):
        if target in anomaly_target:
            num = idx + 1
            continue
    print(target, ' Top K: ', num)
    return num


if __name__ == '__main__':
    
    # Tuning parameters
    alpha = 0.55  
    ad_threshold = 0.045  
    
    
#     folders = ['1', '2', '3', '4', '5']
#     faults_type = ['svc_latency', 'service_cpu', 'service_memory'] #, 'service_memory', 'svc_latency'
# #    faults_type = ['svc_latency', 'service_cpu']
#     faults_type = ['1']
    targets = ['front-end', 'catalogue', 'orders', 'user', 'carts', 'payment', 'shipping', 'unknown']
       
#     if target == 'front-end' and fault_type != 'svc_latency':
#                 #'skip front-end for service_cpu and service_memory'
#                 continue 
#     print('target:', target, ' fault_type:', fault_type)
    
    # prefix of csv files 
    # faults_name = '../faults/' + fault_type + '_' + target
    
    # faults_name = './faults/1/svc_latency/catalogue'
    
    faults_name = './MicroRCA_Online/catalogue'
    latency_df = rt_invocations(faults_name)
    
    # if (target == 'payment' or target  == 'shipping') and fault_type != 'svc_latency':
    #     threshold = 0.02
    # else:
    #     threshold = ad_threshold   
    
    # anomaly detection on response time of service invocation
    anomalies = birch_ad_with_smoothing(latency_df, ad_threshold)
    print('\nanomalies:', anomalies)
    
    # get the anomalous service
    anomaly_nodes = []
    for anomaly in anomalies:
        edge = anomaly.split('_')
        anomaly_nodes.append(edge[1])
    
    anomaly_nodes = set(anomaly_nodes)
    
#                print(anomaly_nodes)
    
    # construct attributed graph
    DG = attributed_graph(faults_name)

    print('\nDG: ', DG)
    
    anomaly_score = anomaly_subgraph(DG, anomalies, latency_df, faults_name, alpha)

    print('\nanomaly_score:')

    for rank in sorted(anomaly_score, key=lambda x: x[1], reverse=True):
        print(rank)

    # anomaly_score_new = []
    # for anomaly_target in anomaly_score:
    #     print(anomaly_target[0])
    #     anomaly_score_new.append(anomaly_target)

    # for target in targets:
    #     num = print_rank(anomaly_score_new, target)

    # filename = 'MicroRCA_results.csv'                    
    # with open(filename,'a') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(['folder', 'catalogue', 'svc_latency', anomaly_score]) 
            