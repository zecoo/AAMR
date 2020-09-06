#!/bin/bash

./headless_locust &

echo '----- Locust started -----'
echo '----- Test will be started in 150s ... ----'

function countdown() {
  for i in $(seq 150 -5 5)
  do
    echo -e "${i}s left"
    sleep 5
    wait
  done
}

for MS in 'user' 'catalogue'
do
  countdown

  kubectl apply -f /root/zik/microservices-demo/zik-test/sock-shop/$MS-delay.yaml

  python3 get_latency.py --fault $MS

  kubectl delete -f /root/zik/microservices-demo/zik-test/sock-shop/$MS-delay.yaml
done

wait
echo '---- Test ends ----'
