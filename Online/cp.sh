#!/bin/bash

echo '----- Locust started -----'
echo '----- RCA will be started in 150s ... ----'

function countdown() {
  for i in $(seq 30 -5 5)
  do
    echo -e "${i}s left"
    sleep 5
    wait
  done
}

for MS in 'user' 'catalogue'
do
  countdown
  echo $MS
done

wait
echo '---- Test ends ----'
