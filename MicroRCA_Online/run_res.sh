#!/bin/bash 

kubectl apply -f /root/zik/microservices-demo/zik-test/delay-test.yaml

n=0
while (($n<10))
do
  python3 MicroRCA_online.py --fault catalogue
  n=$((n+1))
  sleep 10
done 

kubectl delete -f /root/zik/microservices-demo/zik-test/delay-test.yaml
