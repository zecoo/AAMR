import os
import time
from itertools import combinations

rca_arr = ['Microscope_online.py']
svc_arr = ['user', 'catalogue']

# rca_arr = ['Microscope_online.py', 'MicroRCA_online.py', 'tRCA_online.py']
# svc_arr = ['user', 'catalogue', 'order', 'payment']

def combine_svc():
  comb_svc = list(combinations(svc_arr, 2))
  svc_list = []
  for svc in comb_svc:
    svc = svc[0] + '+' + svc[1]
    svc_list.append(svc)
  return svc_list

def countdown():
  time_left = 180
  while time_left > 0:
    print('left: %s s' % time_left)
    time.sleep(5)
    time_left = time_left - 5

if __name__ == '__main__':

  os.system('./headless_locust.sh &')
  print('----- Locust started -----')
  print('----- RCA will be started in 3min ... ----')

  svc_list = combine_svc()
  for svcs in svc_list:
    countdown()

    svc2 = svcs.split('+')
    for svc in svc2:
      os.system('kubectl apply -f /root/zik/fault-injection/sock-shop/%s-delay.yaml' % svc)

    n = 0
    while (n<10):
      for rca in rca_arr:
        os.system('python3 %s --fault %s &' % (rca, svcs))
        time.sleep(10)
      n = n + 1
    
    for svc in svc2:
      os.system('kubectl apply -f /root/zik/fault-injection/sock-shop/%s-delay.yaml' % svc)
    
  print('---- Test ends ----')
