import os
import time
import pandas as pd
import threading
from itertools import combinations

# rca_arr = ['Microscope_online.py']
# svc_arr = ['cartservice', 'productcatalogservice']

rca_arr = ['Microscope_online.py', 'MicroRCA_online.py', 'tRCA_online.py']
svc_arr = ['adservice', 'frontend', 'paymentservice', 'currencyservice', 'cartservice', 'productcatalogservice', 'checkoutservice', 'recommendationservice']
down_time = 180
fault_apply_path = 'kubectl apply -f /root/zik/fault-injection/hipster/'
fault_delete_path = 'kubectl delete -f /root/zik/fault-injection/hipster/'


def combine_svc():
    comb_svc = list(combinations(svc_arr, 2))
    svc_list = []
    for svc in comb_svc:
        svc = svc[0] + '+' + svc[1]
        svc_list.append(svc)
    return svc_list


def tRCA(rca_types, svc):
    global timer
    timer = threading.Timer(5, tRCA, (rca_types, svc))
    for rca in rca_types:
        os.system('python3 %s --fault %s &' % (rca, svc))
        time.sleep(5)
    timer.start()


def countdown(t):
    time_left = t
    while time_left > 0:
        print('left: %s s' % time_left)
        time.sleep(2)
        time_left = time_left - 2


if __name__ == '__main__':

    os.system('kubectl apply -f /root/zik/fault-injection/hipster/adservice.yaml')
    os_str = 'time python3 %s --fault %s' % (rca_arr[0], svc_arr[0])
    f = os.popen(os_str, 'r')
    res = f.readlines()

    for i in res:
        print('------')
        print(i)

    f.close()
    os.system('kubectl delete -f /root/zik/fault-injection/hipster/adservice.yaml')
