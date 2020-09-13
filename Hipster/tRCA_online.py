#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: li
"""

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

from sklearn.cluster import Birch
from sklearn import preprocessing
from numpy import mean

#import seaborn as sns

## =========== Data collection ===========

metric_step = '5s'
smoothing_window = 12

# kubectl get nodes -o wide | awk -F ' ' '{print $1 " : " $6":9100"}'
node_dict = {
                # 'kubernetes-minion-group-103j' : '10.166.0.21:9100',
                # 'kubernetes-minion-group-k2nz' : '10.166.15.235:9100',
                # 'kubernetes-minion-group-kvcr' : '10.166.0.13:9100',
                # 'kubernetes-minion-group-r23j' : '10.166.0.14:9100',
                'iz8vbhflpp3tuw05qfowaxz' : '39.100.0.61:9100'
        }


        

def latency_source_50(prom_url, start_time, end_time, faults_name):

    latency_df = pd.DataFrame()

    # print(start_time)
    # print(end_time)

    response = requests.get(prom_url,
                            params={'query': 'histogram_quantile(0.50, sum(irate(istio_request_duration_seconds_bucket{reporter=\"source\", destination_workload_namespace=\"hipster\"}[1m])) by (destination_workload, source_workload, le))',
                                    'start': start_time,
                                    'end': end_time,
                                    'step': metric_step})
    # results 长这个样子：
    # [{'metric': {'destination_workload': 'orders-db', 'source_workload': 'orders'}, 'value': [1594888889.714, '0.03426666666666667']}, 
    # 解读：value 的第一个值表示当前时间，第二个值表示真正的 value 也就是这一长串 promQL 的 value
    results = response.json()['data']['result']

    # print(results)

    for result in results:
        dest_svc = result['metric']['destination_workload']
        src_svc = result['metric']['source_workload']
        name = src_svc + '_' + dest_svc

        # print(name)

        values = result['values']

        values = list(zip(*values))
        # if 'timestamp' not in latency_df:
        #     timestamp = values[0]
        #     latency_df['timestamp'] = timestamp
        #     latency_df['timestamp'] = latency_df['timestamp'].astype('datetime64[s]')
        metric = values[1]
        latency_df[name] = pd.Series(metric)
        latency_df[name] = latency_df[name].astype('float64')  * 1000

    response = requests.get(prom_url,
                            params={'query': 'sum(irate(istio_tcp_sent_bytes_total{reporter=\"source\",destination_workload_namespace=\"hipster\"}[1m])) by (destination_workload, source_workload) / 1000',
                                    'start': start_time,
                                    'end': end_time,
                                    'step': metric_step})
    results = response.json()['data']['result']

    for result in results:
        dest_svc = result['metric']['destination_workload']
        src_svc = result['metric']['source_workload']
        name = src_svc + '_' + dest_svc
#        print(svc)
        values = result['values']

        values = list(zip(*values))
        # if 'timestamp' not in latency_df:
        #     timestamp = values[0]
        #     latency_df['timestamp'] = timestamp
        #     latency_df['timestamp'] = latency_df['timestamp'].astype('datetime64[s]')
        metric = values[1]
        latency_df[name] = pd.Series(metric)
        latency_df[name] = latency_df[name].astype('float64').rolling(window=smoothing_window, min_periods=1).mean()

    # 这里 df.toscv 写入 scv 文件
    filename = faults_name + '_latency_source_50.csv'
    # latency_df.set_index('timestamp')

    # print('\nlatency_df:')
    # print(latency_df)

    latency_df.to_csv(filename)
    return latency_df


def latency_destination_50(prom_url, start_time, end_time, faults_name):

    latency_df = pd.DataFrame()

    response = requests.get(prom_url,
                            params={'query': 'histogram_quantile(0.50, sum(irate(istio_request_duration_seconds_bucket{reporter=\"destination\", destination_workload_namespace=\"hipster\"}[1m])) by (destination_workload, source_workload, le))',
                                    'start': start_time,
                                    'end': end_time,
                                    'step': metric_step})
    results = response.json()['data']['result']
    # print(results)

    for result in results:
        dest_svc = result['metric']['destination_workload']
        src_svc = result['metric']['source_workload']
        name = src_svc + '_' + dest_svc
        values = result['values']

        values = list(zip(*values))
        # if 'timestamp' not in latency_df:
        #     timestamp = values[0]
        #     latency_df['timestamp'] = timestamp
        #     latency_df['timestamp'] = latency_df['timestamp'].astype('datetime64[s]')
        metric = values[1]
        latency_df[name] = pd.Series(metric)
        latency_df[name] = latency_df[name].astype('float64')  * 1000


    response = requests.get(prom_url,
                            params={'query': 'sum(irate(istio_tcp_sent_bytes_total{reporter=\"destination\",destination_workload_namespace=\"hipster\"}[1m])) by (destination_workload, source_workload) / 1000',
                                    'start': start_time,
                                    'end': end_time,
                                    'step': metric_step})
    results = response.json()['data']['result']

    for result in results:
        dest_svc = result['metric']['destination_workload']
        src_svc = result['metric']['source_workload']
        name = src_svc + '_' + dest_svc
#        print(svc)
        values = result['values']

        values = list(zip(*values))
        # if 'timestamp' not in latency_df:
        #     timestamp = values[0]
        #     latency_df['timestamp'] = timestamp
        #     latency_df['timestamp'] = latency_df['timestamp'].astype('datetime64[s]')
        metric = values[1]
        latency_df[name] = pd.Series(metric)
        latency_df[name] = latency_df[name].astype('float64').rolling(window=smoothing_window, min_periods=1).mean()

    filename = faults_name + '_latency_destination_50.csv'
    # latency_df.set_index('timestamp')
    latency_df.to_csv(filename)
    return latency_df

# 获取 CPU mem network 等系统层面的 metric
# 但是感觉后面没有用到系统层面的 scv 文件啊
def svc_metrics(prom_url, start_time, end_time, faults_name):
    response = requests.get(prom_url,
                            params={'query': 'sum(rate(container_cpu_usage_seconds_total{namespace="hipster", container_name!~\'POD|istio-proxy|\'}[1m])) by (pod_name, instance, container_name)',
                                    'start': start_time,
                                    'end': end_time,
                                    'step': metric_step})
    results = response.json()['data']['result']

    # print(results)

    for result in results:
        df = pd.DataFrame()
        svc = result['metric']['container_name']
        pod_name = result['metric']['pod_name']
        nodename = result['metric']['instance']

        # print(svc)
        values = result['values']

        svc = pod_name.split('-')[0]

        values = list(zip(*values))
        if 'timestamp' not in df:
            timestamp = values[0]
            df['timestamp'] = timestamp
            df['timestamp'] = df['timestamp'].astype('datetime64[s]')
        metric = pd.Series(values[1])
        df['ctn_cpu'] = metric
        df['ctn_cpu'] = df['ctn_cpu'].astype('float64')

        df['ctn_network'] = ctn_network(prom_url, start_time, end_time, pod_name)
        df['ctn_network'] = df['ctn_network'].astype('float64')
        df['ctn_memory'] = ctn_memory(prom_url, start_time, end_time, pod_name)
        df['ctn_memory'] = df['ctn_memory'].astype('float64')

#        response = requests.get('http://localhost:9090/api/v1/query',
#                                params={'query': 'sum(node_uname_info{nodename="%s"}) by (instance)' % nodename
#                                        })
#        results = response.json()['data']['result']
#
#        print(results)
#
#        instance = results[0]['metric']['instance']
        instance = node_dict[nodename]

        # 这里用到了各种的系统层面 metric 
        df_node_cpu = node_cpu(prom_url, start_time, end_time, instance)

        # print(df_node_cpu)
        df = pd.merge(df, df_node_cpu, how='left', on='timestamp')

        df_node_network = node_network(prom_url, start_time, end_time, instance)
        df = pd.merge(df, df_node_network, how='left', on='timestamp')

        df_node_memory = node_memory(prom_url, start_time, end_time, instance)
        df = pd.merge(df, df_node_memory, how='left', on='timestamp')
    
        filename = faults_name + '_' + svc + '.csv'
        df.set_index('timestamp')
        df.to_csv(filename)

# ctn: container
def ctn_network(prom_url, start_time, end_time, pod_name):
    response = requests.get(prom_url,
                            params={'query': 'sum(rate(container_network_transmit_packets_total{namespace="hipster", pod_name="%s"}[1m])) / 1000 * sum(rate(container_network_transmit_packets_total{namespace="hipster", pod_name="%s"}[1m])) / 1000' % (pod_name, pod_name),
                                    'start': start_time,
                                    'end': end_time,
                                    'step': metric_step})
    results = response.json()['data']['result']

    values = results[0]['values']

    values = list(zip(*values))
    metric = pd.Series(values[1])
    return metric


def ctn_memory(prom_url, start_time, end_time, pod_name):
    response = requests.get(prom_url,
                            params={'query': 'sum(rate(container_memory_working_set_bytes{namespace="hipster", pod_name="%s"}[1m])) / 1000 ' % pod_name,
                                    'start': start_time,
                                    'end': end_time,
                                    'step': metric_step})
    results = response.json()['data']['result']

    values = results[0]['values']

    values = list(zip(*values))
    metric = pd.Series(values[1])
    return metric


def node_network(prom_url, start_time, end_time, instance):
    response = requests.get(prom_url,
                            params={'query': 'rate(node_network_transmit_packets_total{device="eth0", instance="%s"}[1m]) / 1000' % instance,
                                    'start': start_time,
                                    'end': end_time,
                                    'step': metric_step})
    results = response.json()['data']['result']
    values = results[0]['values']

    values = list(zip(*values))
    df = pd.DataFrame()
    df['timestamp'] = values[0]
    df['timestamp'] = df['timestamp'].astype('datetime64[s]')
    df['node_network'] = pd.Series(values[1])
    df['node_network'] = df['node_network'].astype('float64')
#    return metric
    return df

def node_cpu(prom_url, start_time, end_time, instance):
    response = requests.get(prom_url,
                            params={'query': 'sum(rate(node_cpu_seconds_total{mode != "idle",  mode!= "iowait", mode!~"^(?:guest.*)$", instance="%s" }[1m])) / count(node_cpu_seconds_total{mode="system", instance="%s"})' % (instance, instance),
                                    'start': start_time,
                                    'end': end_time,
                                    'step': metric_step})
    results = response.json()['data']['result']

    # print(results)
    values = results[0]['values']
    values = list(zip(*values))
#    metric = values[1]
#    print(instance, len(metric))
#    print(values[0])
    df = pd.DataFrame()
    df['timestamp'] = values[0]
    df['timestamp'] = df['timestamp'].astype('datetime64[s]')
    df['node_cpu'] = pd.Series(values[1])
    df['node_cpu'] = df['node_cpu'].astype('float64')
#    return metric
    return df

def node_memory(prom_url, start_time, end_time, instance):
    response = requests.get(prom_url,
                            params={'query': '1 - sum(node_memory_MemAvailable_bytes{instance="%s"}) / sum(node_memory_MemTotal_bytes{instance="%s"})' % (instance, instance),
                                    'start': start_time,
                                    'end': end_time,
                                    'step': metric_step})
    results = response.json()['data']['result']
    values = results[0]['values']

    values = list(zip(*values))
#    metric = values[1]
#    return metric
    df = pd.DataFrame()
    df['timestamp'] = values[0]
    df['timestamp'] = df['timestamp'].astype('datetime64[s]')
    df['node_memory'] = pd.Series(values[1])
    df['node_memory'] = df['node_memory'].astype('float64')
#    return metric
    return df

# Create Graph
# mpg.scv 的构造过程

def mpg(prom_url, faults_name):
    DG = nx.DiGraph()
    df = pd.DataFrame(columns=['source', 'destination'])
    response = requests.get(prom_url,
                            params={'query': 'sum(istio_tcp_received_bytes_total{destination_workload_namespace=\"hipster\"}) by (source_workload, destination_workload)'
                                    })
    
    results = response.json()['data']['result']

    # print(results)

    for result in results:
        metric = result['metric']
        source = metric['source_workload']
        destination = metric['destination_workload']
#        print(metric['source_workload'] , metric['destination_workload'] )
        df = df.append({'source':source, 'destination': destination}, ignore_index=True)
        DG.add_edge(source, destination)
        
        DG.nodes[source]['type'] = 'service'
        DG.nodes[destination]['type'] = 'service'

    response = requests.get(prom_url,
                            params={'query': 'sum(istio_requests_total{destination_workload_namespace=\'hipster\'}) by (source_workload, destination_workload)'
                                    })
    results = response.json()['data']['result']

    for result in results:
        metric = result['metric']
        
        source = metric['source_workload']
        destination = metric['destination_workload']
#        print(metric['source_workload'] , metric['destination_workload'] )
        df = df.append({'source':source, 'destination': destination}, ignore_index=True)
        DG.add_edge(source, destination)
        
        DG.nodes[source]['type'] = 'service'
        DG.nodes[destination]['type'] = 'service'

    response = requests.get(prom_url,
                            params={'query': 'sum(container_cpu_usage_seconds_total{namespace="hipster", container_name!~\'POD|istio-proxy\'}) by (instance, container)'
                                    })
    results = response.json()['data']['result']
    for result in results:
        metric = result['metric']
        if 'container' in metric:
            source = metric['container']
            destination = metric['instance']
            df = df.append({'source':source, 'destination': destination}, ignore_index=True)
            DG.add_edge(source, destination)
            
            DG.node[source]['type'] = 'service'
            DG.node[destination]['type'] = 'host'

    filename = faults_name + '_mpg.csv'
##    df.set_index('timestamp')
    df.to_csv(filename)
    return DG

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


# Anomaly Detection
def birch_ad_with_smoothing(latency_df, threshold):
    # anomaly detection on response time of service invocation. 
    # input: response times of service invocations, threshold for birch clustering
    # output: anomalous service invocation
    
    anomalies = []
    for svc, latency in latency_df.iteritems():
        # No anomaly detection in db
        if svc != 'timestamp' and 'Unnamed' not in svc and 'rabbitmq' not in svc and 'db' not in svc:
            latency = latency.rolling(window=smoothing_window, min_periods=1).mean()
            x = np.array(latency)

            # print(x)
            x = np.where(np.isnan(x), 0, x)

            normalized_x = preprocessing.normalize([x])

            X = normalized_x.reshape(-1,1)

#            threshold = 0.05

            brc = Birch(branching_factor=50, n_clusters=None, threshold=threshold, compute_labels=True)
            brc.fit(X)
            brc.predict(X)

            labels = brc.labels_
#            centroids = brc.subcluster_centers_
            n_clusters = np.unique(labels).size
            if n_clusters > 1:
                anomalies.append(svc)
    return anomalies


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

def calc_score(faults_name):
    
    fault = faults_name.replace('./data/', '')

    latency_filename = faults_name + '_latency_source_50.csv'  # inbound
    latency_df_source = pd.read_csv(latency_filename)

    latency_filename = faults_name + '_latency_destination_50.csv' # outbound
    latency_df_destination = pd.read_csv(latency_filename) 

    # 加和 source
    latency_df_source.loc['all'] = latency_df_source.apply(lambda x:x.sum())

    # 加和 destination
    latency_df_destination.loc['all'] = latency_df_destination.apply(lambda x:x.sum())

    # 获取 locust 数据

    locust_filename = 'example_stats_history.csv'

    locust_df = pd.read_csv(locust_filename)

    locust_latency_50 = []
    # print(len(locust_df))
    if (len(locust_df) < 31):
        locust_latency_50 = locust_df['50%'].tolist()
    else:
        locust_latency_50 = locust_df['50%'][-31:].tolist()
    
    locust_latency_50 = np.nan_to_num(locust_latency_50)
    # print('\n50:', locust_latency_50)

    avg_locust_latency = mean(locust_latency_50)
    # print('\navg:', avg_locust_latency)

    df_data = pd.DataFrame(columns=['svc', 'ratio'])

    # ratio 就是 source / destination
    df_data = (latency_df_source.loc['all'] / latency_df_destination.loc['all']) * (latency_df_source.loc['all'] + latency_df_destination.loc['all']) / avg_locust_latency

    # latency_df_source.loc['all'].to_csv('source.csv')
    # latency_df_destination.loc['all'].to_csv('destination.csv')

    df_data.to_csv('%s_latency_ratio.csv'%faults_name, index=[0])
    # print('\ndf_data: ', df_data)

    ratio = df_data.to_dict()
    trace_based_ratio = {}
    scores = {}

    # print('\nindex: ')
    index  = df_data.index.values

    DG = attributed_graph(faults_name)

    # print('\nkeys: ')

    # 将 ratio 对应到具体的服务
    for key in list(ratio.keys()):
        if 'db' in key or 'rabbitmq' in key or 'Unnamed' in key:
            continue
        else:
            svc_name = key.split('_')[1]
            trace_based_ratio.update({svc_name: ratio[key]})
    
    # print('\ntrace_based_ratio: ', trace_based_ratio)

    # 添加 trace 信息
    # print('\nget trace: ')
    for path in nx.all_simple_paths(DG, source='frontend', target=fault):
        for i in list(itertools.combinations(path, 2)):
            single_trace = i[0] + '_' + i[1]
            if single_trace in index and fault not in single_trace:
                trace_based_ratio[fault] = trace_based_ratio[fault] + ratio[single_trace]

    # 获取邻居个数
    # print('\ndegree: ', DG.degree)
    up = pd.DataFrame(trace_based_ratio, index=[0]).T
    down  = pd.DataFrame(dict(DG.degree), index=[0]).T
    score = (up / down).dropna().to_dict()
    score = score[0]

    # print('\nscore:', score)

    # score 和 服务 进行对应
    score_list = []
    for svc in score:
        item = (svc, score[svc])
        score_list.append(score[svc])

    score_arr = np.array(score_list)

    # print('\nscore_arr: ', score_arr)

    # 归一化处理
    z_score = []
    for x in score_arr:
        x = float(x - score_arr.mean())/score_arr.std() + 0.5
        z_score.append(x)
    
    # print('\nz_score: ', z_score)

    n = 0
    for svc in score:
        score.update({svc: z_score[n]})
        n = n + 1

    # print('\nnew score: ',score)

    return score

def svc_personalization(svc, anomaly_graph, baseline_df, faults_name):

    # 这里用了系统层面 metric 
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

    # print('\nanomaly graph: ', anomaly_graph.adj)

    personalization = calc_score(faults_name)
    # print('\npersonalization: ', personalization)

    anomaly_score = nx.pagerank(DG, alpha=0.85, personalization=personalization, max_iter=10000)

    anomaly_score = sorted(anomaly_score.items(), key=lambda x: x[1], reverse=True)

#    return anomaly_graph
    return anomaly_score


def parse_args():
    """Parse the args."""
    parser = argparse.ArgumentParser(
        description='Root cause analysis for microservices')

    parser.add_argument('--fault', type=str, required=False,
                        default='checkoutservice',
                        help='folder name to store csv file')
    
    # 150s 每隔 5s 取一次数据 所以 csv 文件里一共有 30 行
    # parser.add_argument('--length', type=int, required=False,
    #                 default=150,
    #                 help='length of time series')

    # parser.add_argument('--url', type=str, required=False,
    #                 default='http://http://39.100.0.61:30598/api/v1/query',
    #                 help='url of prometheus query')

    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()
    faults_name = './data/' + args.fault
    filename = ''

    if '+' in faults_name:
        filename = './results/f2/tRCA_results.csv'
    else:
        filename = './results/f1/tRCA_results.csv'

    len_second = 150
    prom_url = 'http://39.100.0.61:32644/api/v1/query_range'
    prom_url_no_range = 'http://39.100.0.61:32644/api/v1/query'
    
    end_time = time.time()
    start_time = end_time - len_second

    # Tuning parameters
    alpha = 0.55  
    ad_threshold = 0.045


    latency_df_source = latency_source_50(prom_url, start_time, end_time, faults_name)
    latency_df_destination = latency_destination_50(prom_url, start_time, end_time, faults_name)
    latency_df = latency_df_destination.add(latency_df_source)

    # print(latency_df)

    svc_metrics(prom_url, start_time, end_time, faults_name)
    DG = mpg(prom_url_no_range, faults_name)

    # anomaly detection on response time of service invocation
    anomalies = birch_ad_with_smoothing(latency_df, ad_threshold)

    if len(anomalies) != 0:
        anomaly_score = anomaly_subgraph(DG, anomalies, latency_df, faults_name, alpha)
        print('\ntRCA_score: ', anomaly_score)

        anomaly_score_new = []
        for anomaly_target in anomaly_score:
            node = anomaly_target[0]
    #       print(anomaly_target[0])
            if DG.nodes[node]['type'] == 'service':
                anomaly_score_new.append(anomaly_target)
        # print(anomaly_score_new)

        fault = faults_name.replace('./data/', '')            
        with open(filename,'a') as f:
            writer = csv.writer(f)
            localtime = time.asctime( time.localtime(time.time()) )
            writer.writerow([localtime, fault, 'svc_latency', anomaly_score_new])
    else:
        print('no anomaly')
