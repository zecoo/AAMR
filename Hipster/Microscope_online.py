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
from scipy.stats import pearsonr

from sklearn.cluster import Birch
from sklearn import preprocessing
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
                            params={'query': 'sum(irate(istio_tcp_sent_bytes_total{reporter=\"source\"}[1m])) by (destination_workload, source_workload) / 1000',
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

    # print('latency_df:')
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
                            params={'query': 'sum(irate(istio_tcp_sent_bytes_total{reporter=\"destination\"}[1m])) by (destination_workload, source_workload) / 1000',
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
                            params={'query': 'sum(istio_tcp_received_bytes_total) by (source_workload, destination_workload)'
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
        # 这里用到了 系统层面的 metric
        max_corr, col = svc_personalization(node, anomaly_graph, baseline_df, faults_name)
        personalization[node] = max_corr / anomaly_graph.degree(node)
#        print(node, personalization[node])

    anomaly_graph = anomaly_graph.reverse(copy=True)
#
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

    anomaly_score = nx.pagerank(anomaly_graph, alpha=0.85, personalization=personalization, max_iter=10000)

    anomaly_score = sorted(anomaly_score.items(), key=lambda x: x[1], reverse=True)

#    return anomaly_graph
    return anomaly_score


def calc_sim(faults_name):
    fault = faults_name.replace('./Online/data/','')

    latency_filename = faults_name + '_latency_source_50.csv'  # inbound
    latency_df_source = pd.read_csv(latency_filename)
    # print("\nfilename")
    # print(latency_filename)

    latency_filename = faults_name + '_latency_destination_50.csv' # outbound
    latency_df_destination = pd.read_csv(latency_filename) 

    # 这里的 fill_value=0 很关键，把 unknown-fe 的 nan 给替换了
    latency_df = latency_df_source.add(latency_df_destination, fill_value=0)

    # print('\nlatency_df: ')
    print(latency_df)
    latency_df.to_csv('%s_latency.csv'%fault)

    # 获取 locust 数据
    locust_filename = './example_stats_history.csv'
    locust_df = pd.read_csv(locust_filename)
    print(locust_df)

    locust_latency_50 = []
    # print(len(locust_df))
    if (len(locust_df) < 31):
        locust_latency_50 = locust_df['50%'].tolist()
    else:
        locust_latency_50 = locust_df['50%'][-31:].tolist()
    
    locust_latency_50 = np.nan_to_num(locust_latency_50)
    # print('\n50:', locust_latency_50)

    fklen = len(locust_latency_50)
    # print('\nfklen: ', fklen)

    svc_latency_df = pd.DataFrame()

    for key in latency_df.keys():
        if 'db' in key or 'rabbitmq' in key or 'Unnamed' in key:
            continue
        else:
            svc_name = key.split('_')[1]
            if svc_name in svc_latency_df:
                svc_latency_df[svc_name].add(latency_df[key])
            else:
                svc_latency_df[svc_name] = latency_df[key]

    # print('\nsvc_len:', len(svc_latency_df))

    score = {}
    for key in svc_latency_df.keys():
        # len : 31
        # print(len(svc_latency_df[key].tolist()))

        # 输出:(r, p)
        # r:相关系数[-1，1]之间
        # p:p值越小
        score.update({key: pearsonr(svc_latency_df[key].tolist()[-fklen:], locust_latency_50)[0]})
    
    score = sorted(score.items(), key = lambda kv:(kv[1], kv[0]), reverse=True)
    # print(score)

    corr_df = latency_df.corr()
    # corr_df[corr_df] = np.nan
    # print('\ncorr: ', corr_df)

    corr_df.to_csv('%s_corr.csv'%fault)

    return score

def parse_args():
    """Parse the args."""
    parser = argparse.ArgumentParser(
        description='Root cause analysis for microservices')

    parser.add_argument('--fault', type=str, required=False,
                        default='adservice',
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
    svc_metrics(prom_url, start_time, end_time, faults_name)

    filename = './results/Microscope_results.csv'
    fault = faults_name.replace('./data/', '')
    rank = calc_sim(faults_name)
    print('\nMicroscope Score:', rank)
    with open(filename,'a') as f:
        writer = csv.writer(f)
        localtime = time.asctime( time.localtime(time.time()) )
        writer.writerow([localtime, fault, 'svc_latency', rank])
