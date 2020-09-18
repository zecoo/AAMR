import pandas as pd
from datetime import datetime
import time

svc_arr = ['carts', 'shipping', 'user', 'catalogue', 'orders', 'payment']
rca_arr = ['Microscope_online.py', 'MicroRCA_online.py', 'tRCA_online.py']
time_arr = ['time_ms.csv', 'time_tRCA.csv', 'time_mRCA.csv']

def str2time(str):
        return float(str.split(':')[2])

for res in time_arr:
    print('=============')
    print(res)
    print('=============')
    time_df = pd.read_csv('results/%s' % res)

    time_df.columns = ['ts', 'svc', 'time']
    for svc in svc_arr:
        svc_time = 0
        svc_num = 0

        for row in time_df.itertuples():
            time0 = str2time(getattr(row, 'time'))
            if getattr(row, 'svc') == svc:
                if time0 < 0 or time0 > 50:
                    pass
                else:
                    svc_time = time0 + svc_time
                    svc_num = svc_num + 1

        if svc_num != 0:
            svc_time = svc_time / svc_num
            print(svc + ": " + str(svc_time))