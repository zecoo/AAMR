import numpy as np
from numpy import mean
from dtw import dtw
from itertools import combinations
import pandas as pd

# We define two sequences x, y as numpy array
# where y is actually a sub-sequence from x

list1 = [5.044463702, 5.462427864, 5.601749252, 5.671409945, 5.454981289, 5.310695519, 5.207634254, 5.362579973, 5.483093309, 5.579503978, 5.537791495, 5.503031093, 5.50938134, 5.47818756, 5.44699378, 5.4158, 5.495811111, 5.575822222, 5.655833333, 5.474411111, 5.292988889, 5.111566667, 5.097188889, 5.082811111, 5.068433333, 5.092416667, 5.1164, 5.140383333, 5.241944444, 5.343505556, 5.445066667]
list2 = [300, 300, 300, 200, 300, 2.711076102, 2.768112849, 2.848523743, 2.911065549, 2.961098994, 2.96693848, 2.971804717, 2.96572193, 2.974336842, 2.982951755, 2.991566667, 3.001777778, 3.011988889, 3.0222, 3.014438889, 3.006677778, 2.998916667, 2.988777778, 2.978638889, 2.9685, 2.984266667, 3.000033333, 3.0158, 3.050494444, 3.085188889, 3.119883333]
list0 = [34.46896874, 47.55267225, 51.91390675, 54.094524, 49.7155792, 46.79628267, 44.71107086, 46.605387, 48.078744, 49.2574296, 47.69276024, 46.38886911, 46.18696061, 46.17810152, 46.16924242, 46.16038333, 46.13442778, 46.10847222, 46.08251667, 45.32696667, 44.57141667, 43.81586667, 45.00411667, 46.19236667, 47.38061667, 46.10135556, 44.82209444, 43.54283333, 44.77881111, 46.01478889, 47.25076667]

svc_arr = ['frontend', 'paymentservice', 'currencyservice', 'cartservice', 'productcatalogservice', 'checkoutservice', 'recommendationservice', 'emailservice']

def pre_score(data):
	res = []
	for i in range(0, len(data)):
		if i == (len(data)-1):
			res.append(data[i] - mean(data))
		else:
			res.append(data[i+1] - data[i])
	return res

def cal(x, y):
	x = np.array(pre_score(x)).reshape(-1, 1)
	y = np.array(pre_score(y)).reshape(-1, 1)

	manhattan_distance = lambda x, y: np.abs(x - y)

	d, cost_matrix, acc_cost_matrix, path = dtw(x, y, dist=manhattan_distance)

	return d

def get_svc_latency_df(faults_name):
	fault = faults_name.replace('./data/','')
	
	latency_filename = faults_name + '_latency_source_50.csv'
	latency_df_source = pd.read_csv(latency_filename)
    # print("\nfilename")
    # print(latency_filename)

	latency_filename = faults_name + '_latency_destination_50.csv'
	latency_df_destination = pd.read_csv(latency_filename) 

	# 这里的 fill_value=0 很关键，把 unknown-fe 的 nan 给替换了
	latency_df = latency_df_source.add(latency_df_destination, fill_value=0)

	# print('\nlatency_df: ')
	latency_len = len(latency_df)

	svc_latency_df = pd.DataFrame()

	for key in latency_df.keys():
		if 'db' in key or 'rabbitmq' in key or 'Unnamed' in key:
			continue
		else:
			svc_name = key.split('_')[1]
			if svc_name in svc_latency_df:
				svc_latency_df[svc_name].add(latency_df[key])
			else:
				svc_latency_df[svc_name] = latency_df[key]

	return svc_latency_df

def anomaly_detection(faults_name):

	has_anomaly = False
	
	svc_latency_df = get_svc_latency_df(faults_name)
	svc_latency_df = svc_latency_df.fillna(svc_latency_df.mean())

	for svc in svc_arr:
		x = svc_latency_df['frontend']
		y = svc_latency_df[svc]

		if cal(x,y) > 1000:
			print('AAAAAAAAnomaly')
			has_anomaly = True
	
	return has_anomaly

if __name__ == '__main__':
	faults_name = './data/' + 'currencyservice'
	anomaly_detection(faults_name)