#!/bin/bash

var='MicroRCA_online.py'

./headless_locust.sh &

echo '----- Locust started -----'
echo '----- RCA will be started in 3min ... ----'

function countdown() {
  for i in $(seq 180 -5 5)
  do
    echo -e "${i}s left"
    sleep 5
    wait
  done
}

for MS in 'user,catalogue' 'catalogue,payment'
do
  countdown

  arr=(${str//,/ })

  for i in ${arr[@]}  
  do  
      echo $i 
      kubectl apply -f /root/zik/fault-injection/sock-shop/$i-delay.yaml
  done

  n=0
  while (($n<10))
  do
    python3 $var --fault $MS &
    n=$((n+1))
    sleep 10
  done

  for i in ${arr[@]}  
  do  
      echo $i 
      kubectl delete -f /root/zik/fault-injection/sock-shop/$i-delay.yaml
  done

done

wait
echo '---- Test ends ----'
