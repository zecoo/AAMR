#!/bin/bash

function countdown() {
  for i in $(seq 150 -10 10)
  do
    echo -e "$i s left"
    sleep 10
    wait
  done
}

echo 'RCA starting...'

# for MS in 'user' 'catalogue' 'order' 'shipping' 'payment' 'carts'
for MS in 'user' 'catalogue'
do
  countdown

  kubectl apply -f /root/zik/microservices-demo/zik-test/sock-shop/$MS-delay.yaml

  n=0
  while (($n<10))
  do
    python3 tRCA_online.py --fault $MS &
    python3 MicroRCA_online.py --fault $MS &
    python3 Microscope_online.py --fault $MS &
    n=$((n+1))
    sleep 10
  done 

  kubectl delete -f /root/zik/microservices-demo/zik-test/sock-shop/$MS-delay.yaml
done

exit
