echo '----- Locust started -----'

locust --host="http://39.100.0.61:30001" -u 50 -r 20 --headless --run-time 60m --csv=example 1>/dev/null 2>/dev/null & 

exit
