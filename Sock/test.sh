#!/bin/bash

var='Microscope_online.py'

./headless_locust.sh &

echo '----- Locust started -----'
echo '----- RCA will be started in 150s ... ----'

function countdown() {
  for i in $(seq 180 -5 5)
  do
    echo -e "${i}s left"
    sleep 5
    wait
  done
}

for MS in 'user' 'catalogue' 'orders' 'payment'
do
  countdown

  kubectl apply -f /root/zik/fault-injection/sock-shop/$MS-delay.yaml

  n=0
  while (($n<10))
  do
    python3 $var --fault $MS &
    n=$((n+1))
    sleep 10
  done

  kubectl delete -f /root/zik/fault-injection/sock-shop/$MS-delay.yaml
done

wait
echo '---- Test ends ----'
