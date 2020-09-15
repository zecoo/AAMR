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

from sklearn.cluster import Birch
from sklearn import preprocessing
from functools import wraps
#import seaborn as sns

## =========== Data collection ===========

metric_step = '5s'
smoothing_window = 12

prom_url = 'http://39.100.0.61:31423/api/v1/query_range'
prom_url_no_range = 'http://39.100.0.61:31423/api/v1/query'

# kubectl get nodes -o wide | awk -F ' ' '{print $1 " : " $6":9100"}'
node_dict = {
                # 'kubernetes-minion-group-103j' : '10.166.0.21:9100',
                # 'kubernetes-minion-group-k2nz' : '10.166.15.235:9100',
                # 'kubernetes-minion-group-kvcr' : '10.166.0.13:9100',
                # 'kubernetes-minion-group-r23j' : '10.166.0.14:9100',
                'iz8vbhflpp3tuw05qfowaxz' : '39.100.0.61:9100'
        }

def fn_timer(function):
    @wraps(function)
    def function_timer(*args, **kwargs):
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        print ("Total time running %s: %s seconds" %
            (function.__name__, str(t1-t0))
            )
        return result
    return function_timer
        
# 获取 CPU mem network 等系统层面的 metric
# 但是感觉后面没有用到系统层面的 scv 文件啊
@fn_timer
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
        # svc = result['metric']['container_name']
        pod_name = result['metric']['pod_name']
        nodename = result['metric']['instance']

        if len(pod_name.split('-')) > 3:
            svc = pod_name.split('-')[0] + '-' + pod_name.split('-')[1]
        else:
            svc = pod_name.split('-')[0]

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
@fn_timer
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
    
    end_time = time.time()
    start_time = end_time - len_second

    svc_metrics(prom_url, start_time, end_time, faults_name)
    DG = mpg(prom_url_no_range, faults_name)