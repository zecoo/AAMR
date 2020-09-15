import numpy as np
from numpy import mean
from dtw import dtw

# We define two sequences x, y as numpy array
# where y is actually a sub-sequence from x

list1 = [5.044463702, 5.462427864, 5.601749252, 5.671409945, 5.454981289, 5.310695519, 5.207634254, 5.362579973, 5.483093309, 5.579503978, 5.537791495, 5.503031093, 5.50938134, 5.47818756, 5.44699378, 5.4158, 5.495811111, 5.575822222, 5.655833333, 5.474411111, 5.292988889, 5.111566667, 5.097188889, 5.082811111, 5.068433333, 5.092416667, 5.1164, 5.140383333, 5.241944444, 5.343505556, 5.445066667]
list2 = [300, 300, 300, 200, 300, 2.711076102, 2.768112849, 2.848523743, 2.911065549, 2.961098994, 2.96693848, 2.971804717, 2.96572193, 2.974336842, 2.982951755, 2.991566667, 3.001777778, 3.011988889, 3.0222, 3.014438889, 3.006677778, 2.998916667, 2.988777778, 2.978638889, 2.9685, 2.984266667, 3.000033333, 3.0158, 3.050494444, 3.085188889, 3.119883333]
list0 = [34.46896874, 47.55267225, 51.91390675, 54.094524, 49.7155792, 46.79628267, 44.71107086, 46.605387, 48.078744, 49.2574296, 47.69276024, 46.38886911, 46.18696061, 46.17810152, 46.16924242, 46.16038333, 46.13442778, 46.10847222, 46.08251667, 45.32696667, 44.57141667, 43.81586667, 45.00411667, 46.19236667, 47.38061667, 46.10135556, 44.82209444, 43.54283333, 44.77881111, 46.01478889, 47.25076667]

lists = [list1, list2, list0]

list3 = [1,2,3,4]
list4 = []

def pre_score(data):
	res = []
	for i in range(0, len(data)):
		if i == (len(data)-1):
			res.append(data[i] - mean(data))
		else:
			res.append(data[i+1] - data[i])
	return res

def standardization(data):
	mu = np.mean(data, axis=0)
	sigma = np.std(data, axis=0)
	return (data - mu) / sigma


def normalization(data):
  _range = np.max(data) - np.min(data)
  return data / _range

def Z_Score(data):
  lenth = len(data)
  total = sum(data)
  ave = float(total)/lenth
  tempsum = sum([pow(data[i] - ave,2) for i in range(lenth)])
  tempsum = pow(float(tempsum)/lenth,0.5)
  for i in range(lenth):
      data[i] = (data[i] - ave)/tempsum
  return data

# for each in list0:
# 	tmp = each - mean(list0)
# 	list3.append(tmp)

# print(list3)

# for each in list2:
# 	tmp = each - mean(list2)
# 	list4.append(tmp)

def cal(x, y):
	x = np.array(pre_score(x)).reshape(-1, 1)
	y = np.array(pre_score(y)).reshape(-1, 1)

	manhattan_distance = lambda x, y: np.abs(x - y)

	d, cost_matrix, acc_cost_matrix, path = dtw(x, y, dist=manhattan_distance)

	print(d)


def main():
	cal(list1, list2)
	cal(list1, list0)
	cal(list2, list0)

if __name__ == '__main__':
	main()



# import matplotlib.pyplot as plt

# plt.imshow(acc_cost_matrix.T, origin='lower', cmap='gray', interpolation='nearest')
# plt.plot(path[0], path[1], 'w')
# plt.show()