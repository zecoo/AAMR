#!/bin/bash 

locust --host="http://39.100.0.61:30001" -u 100 -r 20 --headless --csv=example 1>/dev/null 2>/dev/null

sleep 50
wait

echo 'finished'

for MS in 'user' 'catalogue' 'order' 'shipping' 'payment' 'carts'

do
  kubectl apply -f /root/zik/microservices-demo/zik-test/sock-shop/$MS-delay.yaml

  n=0
  while (($n<10))
  do
    python3 tRCA_online.py --fault $MS &
    python3 MicroRCA_online.py --fault $MS &
    n=$((n+1))
    sleep 10
  done 

  kubectl delete -f /root/zik/microservices-demo/zik-test/sock-shop/$MS-delay.yaml

  sleep 150
	wait
done


