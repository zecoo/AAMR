#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import seaborn as sns
import matplotlib.pyplot as plt
import ssl
import pandas as pd 
import numpy as np

def result_pic(result, index):
    """
    雷达图的绘制
    :param result: 分类数据
    :return: 雷达图
    """
    # 解析出类别标签和种类
    labels = ['payment', 'currency', 'cart', 'productcatalog', 'checkout', 'recommendation', 'frontend']
    new_labels = ['payment', 'currency', 'cart', 'productcatalog', 'checkout', 'recommendation', 'frontend', 'payment']
    
    
    kinds = list(result.iloc[:, 0])


    # 由于在雷达图中，要保证数据闭合，这里就再添加L列，并转换为 np.ndarray
    result = pd.concat([result, result[['payment']]], axis=1)

    centers = np.array(result.iloc[:, 1:])

    # 分割圆周长，并让其闭合
    n = len(labels)
    angle = np.linspace(0, 2 * np.pi, n, endpoint=False)
    angle = np.concatenate((angle, [angle[0]]))

    # 绘图

    # 10.5/9, 7
    fig = plt.figure(figsize=(10.5, 7))
    ax = fig.add_subplot(111, polar=True)    # 参数polar, 以极坐标的形式绘制图形

    # 画线
    for i in range(len(kinds)):
        ax.plot(angle, centers[i], linewidth=2.5, label=kinds[i])
        ax.fill(angle, centers[i], alpha=0.20) # 填充颜色
        # ax.fill(angle, centers[i])  # 填充底色

    # 添加属性标签
    ax.set_thetagrids(angle * 180 / np.pi, new_labels)

    plt.grid(linestyle='-.')
    # 设置图例字体大小
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    title = ''
    if index == 0:
        title = 'AC@1 for Latency Delay case'
    elif index == 1:
        title = 'AC@1 for CPU Hog case'
    elif index == 2:
        title = 'AC@1 for Container Pause case'
        # !!!!!!!! 图例hiiiiiiin重要
        plt.legend(loc='lower right')
        plt.legend( bbox_to_anchor=(1.05, 0.20), loc=2, borderaxespad=0, numpoints=4, fontsize=15)
    plt.title(title ,fontdict={'weight':'normal','size': 20}, y=1.06)

    filename = './radar' + str(index) + '.pdf'
    plt.savefig(filename ,format='pdf',dpi = 300)
    plt.show()


if __name__ == '__main__':

    for i in range(0,1):
        filename = 'radar' + str(i) + '.csv'
        result = result = pd.read_csv(filename, sep=',')
        result_pic(result, i)