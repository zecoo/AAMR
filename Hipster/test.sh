#!/bin/bash

var='Microscope_online.py'

# ./headless_locust.sh &

echo '----- RCA will be started in 3min ... ----'

function countdown() {
  for i in $(seq 180 -5 5)
  do
    echo -e "${i}s left"
    sleep 5
    wait
  done
}

for MS in 'checkoutservice' 'cartservice'
do
  # countdown

  kubectl apply -f /root/zik/fault-injection/hipster/$MS.yaml

  n=0
  while (($n<10))
  do
    python3 $var --fault $MS &
    n=$((n+1))
    sleep 10
  done
  
  kubectl delete -f /root/zik/fault-injection/hipster/$MS.yaml

done

wait
echo '---- Test ends ----'
