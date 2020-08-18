#!/bin/bash 

for MS in 'user' 'carts' 'catalogue'
do
  kubectl apply -f /root/zik/microservices-demo/zik-test/$MS-delay.yaml

  n=0
  while (($n<10))
  do
    python3 tRCA_online.py --fault $MS
    n=$((n+1))
    sleep 10
  done 

  kubectl delete -f /root/zik/microservices-demo/zik-test/$MS-delay.yaml

  sleep 150
	wait
done


