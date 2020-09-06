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
import os

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
                            params={'query': 'histogram_quantile(0.50, sum(irate(istio_request_duration_seconds_bucket{reporter=\"source\", destination_workload_namespace=\"sock-shop\"}[1m])) by (destination_workload, source_workload, le))',
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
                            params={'query': 'histogram_quantile(0.50, sum(irate(istio_request_duration_seconds_bucket{reporter=\"destination\", destination_workload_namespace=\"sock-shop\"}[1m])) by (destination_workload, source_workload, le))',
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
                            params={'query': 'sum(rate(container_cpu_usage_seconds_total{namespace="sock-shop", container_name!~\'POD|istio-proxy|\'}[1m])) by (pod_name, instance, container_name)',
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

def parse_args():
    """Parse the args."""
    parser = argparse.ArgumentParser(
        description='Root cause analysis for microservices')

    parser.add_argument('--fault', type=str, required=False,
                        default='user',
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

    # response = requests.get(prom_url,
    #                         params={'query': 'sum(irate(istio_tcp_sent_bytes_total{reporter=\"source\"}[1m])) by (destination_workload, source_workload) / 1000',
    #                                 'start': start_time,
    #                                 'end': end_time,
    #                                 'step': metric_step})
    # results = response.json()['data']['result']

    # print(results)
    
    for i in range(0, 10):
        faults_name = faults_name + str(i)
        latency_df_source = latency_source_50(prom_url, start_time, end_time, faults_name)
        latency_df_destination = latency_destination_50(prom_url, start_time, end_time, faults_name)
        latency_df = latency_df_destination.add(latency_df_source)
        os.system('sleep 10')
        os.system('wait')
