import os
import time
import pandas as pd
import threading
import random
from itertools import combinations

# rca_arr = ['Microscope_online.py']
# svc_arr = ['cartservice', 'productcatalogservice']

# rca_arr = ['Microscope_online.py', 'MicroRCA_online.py', 'tRCA_online.py']
rca_arr = ['PCA_online.py']
# svc_arr = ['paymentservice', 'currencyservice', 'cartservice', 'productcatalogservice', 'checkoutservice', 'recommendationservice', 'frontend']
svc_arr = ['cartservice']
# random.shuffle(svc_arr)
down_time = 180
fault_apply_path = 'kubectl apply -f /root/zik/fault-injection/hipster/'
fault_delete_path = 'kubectl delete -f /root/zik/fault-injection/hipster/'

replicas = 1
case = 1

def combine_svc(svcs):
    comb_svc = list(combinations(svcs, 2))
    svc_list = []
    for svc in comb_svc:
        svc = svc[0] + '+' + svc[1]
        svc_list.append(svc)
    return svc_list


def tRCA(rca_types, svc):
    global timer
    timer = threading.Timer(4, tRCA, (rca_types, svc))
    for rca in rca_types:
        os.system('python3 %s --fault %s --replicas %d &' % (rca, svc, replicas))
        time.sleep(4)
    timer.start()


def countdown(t):
    time_left = t
    while time_left > 0:
        print('left: %s s' % time_left)
        time.sleep(2)
        time_left = time_left - 2


if __name__ == '__main__':
    # os.system('./headless_locust.sh &')
    print('==== RCA will be started in 3min ... ====')
    if case == 1:
        for svc in svc_arr:
            countdown(down_time)
            os.system(fault_apply_path + '%s.yaml' % svc)
            timer = threading.Timer(5, tRCA, (rca_arr, svc))
            timer.start()
            time.sleep(120)
            timer.cancel()
            os.system(fault_delete_path + '%s.yaml' % svc)
        print("==== ends ====")
    elif case == 2:
        svc_list = combine_svc(svc_arr)
        for svcs in svc_list:
            countdown(down_time)
            svc2 = svcs.split('+')
            # create fault injection
            for svc in svc2:
                # os.system('kubectl apply -f /root/zik/fault-injection/hipster/%s.yaml' % svc)
                os.system(fault_apply_path + '%s.yaml' % svc)
            # interval apply RCA
            timer = threading.Timer(4, tRCA, (rca_arr, svcs))
            timer.start()
            time.sleep(120)
            timer.cancel()
            # delete fault injection
            for svc in svc2:
                # os.system('kubectl delete -f /root/zik/fault-injection/hipster/%s.yaml' % svc)
                os.system(fault_delete_path + '%s.yaml' % svc)

    print('==== Experiment ends ====')
