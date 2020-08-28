#!/bin/bash

./headless_locust.sh &

echo '----- Locust started -----'
echo '----- RCA will be started in 150s ... ----'

code = 'tRCA_online.py'

function countdown() {
  for i in $(seq 100 -10 10)
  do
    echo -e "$i s left"
    sleep 10
    wait
  done
}

for MS in 'user' 'catalogue'

do
  countdown

  kubectl apply -f /root/zik/microservices-demo/zik-test/sock-shop/$MS-delay.yaml

  n=0
  while (($n<5))
  do
    python3 $code --fault $MS 1>testRCA.log 2>testError.log &
    n=$((n+1))
    sleep 10
  done

  kubectl delete -f /root/zik/microservices-demo/zik-test/sock-shop/$MS-delay.yaml

done

wait
echo '---- Test ends ----'
