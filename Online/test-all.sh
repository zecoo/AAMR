#!/bin/bash

./headless_locust.sh &

echo '----- Locust started -----'
echo '----- RCA will be started in 150s ... ----'

./test.sh 1>testRCA.log 2>testError.log &

wait
echo '---- Test ends ----'
