#!/bin/bash

arr=(MicroRCA_online.py Microscope_online.py tRCA_online.py)

./headless_locust.sh &

echo '----- RCA will be started in 150s ... ----'

function countdown() {
  for i in $(seq 150 -5 5)
  do
    echo -e "${i}s left"
    sleep 5
    wait
  done
}

for MS in 'user' 'catalogue' 'order' 'payment' 'carts' 'shipping'
do
  countdown

  kubectl apply -f /root/zik/microservices-demo/zik-test/sock-shop/$MS-delay.yaml

  n=0
  while (($n<5))
  do
    for var in ${array[@]}
    do
      python3 $var --fault $MS &
    done
    n=$((n+1))
    sleep 10
  done

  kubectl delete -f /root/zik/microservices-demo/zik-test/sock-shop/$MS-delay.yaml
done

wait
echo '---- Test ends ----'