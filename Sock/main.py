import os
import time

var = ('Microscope_online.py', 'MicroRCA_online.py')
svc_arr = ('user', 'catalogue')

os.system('./headless_locust.sh &')

print('----- Locust started -----')
print('----- RCA will be started in 3min ... ----')

def countdown():
  time_left = 180
  while time_left > 0:
    print('left: ',time_left)
    time.sleep(5)
    time_left = time_left - 5

if __name__ == '__main__':

  for svc in svc_arr:
    countdown()
    os.system('kubectl apply -f /root/zik/fault-injection/sock-shop/%s-delay.yaml' % svc)

    n = 0
    while (n<10):
      os.system('python3 %s --fault %s &' % (var[0], svc))
      time.sleep(10)
      n = n + 1
    
    os.system('kubectl delete -f /root/zik/fault-injection/sock-shop/%s-delay.yaml' % svc)
    
  print('---- Test ends ----')
