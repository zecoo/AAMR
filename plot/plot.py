import seaborn as sns
import matplotlib.pyplot as plt
import ssl
import pandas as pd 
import numpy as np

ssl._create_default_https_context = ssl._create_unverified_context

sns.set(style="whitegrid")
# 修改折线圆点的类型
markers= ['>', 'o', 'd', 's']
# 获取数据

def str2time(str):
    return float(str.split(':')[1])

def get_line_plot():
    
    plt.figure(figsize=(11, 5.5))
    dataset = pd.read_csv("./Hipster/results/f1/replicas1.csv")
    dataset.columns = ['rca', 'The number of replicas', '1fPR@1', 'AC@1', '2fPR@2', '2fMAP@2']

    # print(dataset)
    ax1 = sns.lineplot(y="2fMAP@2", x="The number of replicas", hue="rca", style='rca', markers=markers, dashes=False, linewidth=2, data=dataset)
    box = ax1.get_position()
    my_y_ticks = np.arange(0, 1.01, 0.1)
    my_x_ticks = np.arange(7, 72.01, 7)

    # 对边框进行加粗
    ax=plt.gca();#获得坐标轴的句柄
    ax.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2);####设置上部坐标轴的粗细
    ax.spines['bottom'].set_color('k');###设置底部坐标轴的粗细
    ax.spines['left'].set_color('k');####设置左边坐标轴的粗细
    ax.spines['right'].set_color('k');###设置右边坐标轴的粗细
    ax.spines['top'].set_color('k');####设置上部坐标轴的粗细
    plt.ylabel('AC@1', fontdict={'weight':'normal','size': 15})
    plt.xlabel('The number of replicas', fontdict={'weight':'normal','size': 15})
    plt.xlim(10.5,72.5)
    plt.ylim(0,1.05)
    plt.xticks(my_x_ticks, fontsize=15)
    plt.yticks(my_y_ticks, fontsize=15)
    # ax1.set_position([box.x0, box.y0, box.width , box.width* 0.8])
    ax1.legend(loc='center left', bbox_to_anchor=(0.12, 1.06),ncol=4, fontsize=14)
    # ax1.legend( bbox_to_anchor=(1.05, 0.98), loc=2, borderaxespad=0, numpoints=1)
    # fig.subplots_adjust(right=0.8)
    plt.savefig('./line.pdf',format='pdf',dpi = 300)
    plt.show()

def get_bar_plot():
    fig, ax1 = plt.subplots()

    dataset = pd.read_csv("results/2ss.csv")
    dataset.columns = ['method', 'value', 'Online-botique']

    ax1 = sns.barplot(y="value", x="Online-botique", hue="method", data=dataset)
    box = ax1.get_position()
    # ax1.set_position([box.x0, box.y0, box.width , box.width* 0.8])
    # ax1.legend(loc='center left', bbox_to_anchor=(0.2, 1.12),ncol=3)
    ax1.legend( bbox_to_anchor=(1.04, 1), loc=2, borderaxespad=0, numpoints=1)
    fig.subplots_adjust(right=0.70)
    plt.savefig('./bar.pdf',format='pdf',dpi = 300)
    plt.show()

def get_box_plot():
    
    fig, ax1 = plt.subplots()
    plt.figure(figsize=(10, 5))
    time_df = pd.read_csv("THipster/results/test.csv")
    time_df.columns = ['time', 'svc', 'value', 'rca']
    
    value_list = []
    for row in time_df.itertuples():
        time0 = str2time(getattr(row, 'value'))
        value_list.append(time0)
    
    time_df.loc[:,('value')] = value_list

    print(time_df)
    ax1 = sns.boxplot(x='svc' , y='value' , hue='rca', data=time_df)
    # box = ax1.get_position()
    # ax1.set_position([box.x0, box.y0, box.width , box.width* 0.8])
    # ax1.legend(loc='center left', bbox_to_anchor=(0.2, 1.12),ncol=3)

    # 这两个
    # ax1.legend( bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0, numpoints=1)
    # fig.subplots_adjust(right=0.8)
    plt.show()

if __name__ == "__main__":
    # get_bar_plot()
    # get_box_plot()
    get_line_plot()