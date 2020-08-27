#!/bin/bash

./headless_locust.sh &

echo ''
echo '----- Locust started -----'
echo '----- RCA will be started in 150s ... ----'

./sock-shop_run.sh 1>RCA.log 2>Error.log &
