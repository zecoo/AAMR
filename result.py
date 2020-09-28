import pandas as pd
import csv
from _tkinter import _flatten


f1_benchmarks = []
for i in range(1,10):
    f1_benchmarks.append('./Hipster/results/f1/%d/' % i)
f2_benchmarks = ['./Hipster/results/f2/']
rca_arr = ['Microscope', 'MicroRCA', 'tRCA']

topK = 1

# 统计所有的类别
def get_unique_labels(y_true, y_pred):
    y_true_set = set(y_true)
    # y_pred_set = set(_flatten(y_pred))
    # unique_label_set = y_true_set | y_pred_set
    # unique_label = list(unique_label_set)
    unique_label = list(y_true_set)
    return unique_label

# y_true: 1d-list-like
# y_pred: 2d-list-like
# k：针对top-k各结果进行计算（k <= y_pred.shape[1]）
def precision_k_f1(y_trues, y_preds, k=topK, digs=2):
    # 取每个样本的top-k个预测结果！
    y_preds = [pred[:k] for pred in y_preds]
    # print(y_preds)

    unique_labels = get_unique_labels(y_trues, y_preds)
    num_classes = len(unique_labels)
    svc_pr1 = 0
    # 计算每个类别的precision、recall、f1-score、support
    results_dict = {}
    results = ''
    for label in unique_labels:
        # print(label)
        current_label_result = []
        # TP + FN
        tp_fn = y_trues.count(label)
        # TP + FP
        tp_fp = 0
        for y_pred in y_preds:
            if label in y_pred:
                tp_fp += 1
        
        # TP
        tp = 0
        for i in range(len(y_trues)):
            if y_trues[i] == label and label in y_preds[i]:
                tp += 1

        support = tp_fn
        # print(label, support, tp)

        try:
            precision = round(tp/support, digs)
            # precision = round(tp/tp_fp, digs)
            recall = round(tp/tp_fn, digs)
            f1_score = round(2*(precision * recall) /
                             (precision + recall), digs)
        except ZeroDivisionError:
            precision = 0
            recall = 0
            f1_score = 0

        current_label_result.append(precision)
        current_label_result.append(support)
        # 输出第一行
        results_dict[str(label)] = current_label_result
    title = '\t' + 'precision@' + str(k) + '\t' + 'num' + '\n'
    results += title

    for k, v in sorted(results_dict.items()):
        if (str(v[1]) == '0'):
            continue
        else:
            current_line = str(
                k) + '\t\t' + str(v[0]) + '\t\t' + str(v[1]) + '\n'
            results += current_line
            svc_pr1 += v[0]
    sums = len(y_trues)

    s_map = round(svc_pr1 / num_classes, digs)


    weighted_avg_results = [(v[0]*v[1]) for k, v in sorted(results_dict.items())]

    # 计算weighted avg
    weighted_precision = 0
    weighted_recall = 0
    weighted_f1_score = 0
    for weighted_avg_result in weighted_avg_results:
        weighted_precision += weighted_avg_result

    weighted_precision /= sums
    weighted_precision = round(weighted_precision, digs)

    weighted_avg_line = 'avg:' + '\t\t' + str(round(weighted_precision, digs)) + '\t\t' + str(sums) + '\n' + 'map:' + '\t\t' + str(s_map)

    results += weighted_avg_line
    print(results)
    print()

    return weighted_precision, s_map

def precision_k_f2(y_trues, y_preds, k=topK, digs=2):
    # 取每个样本的top-k个预测结果！
    y_preds = [pred[:k] for pred in y_preds]
    # print(y_preds)
    svc_pr1 = 0
    unique_labels = get_unique_labels(y_trues, y_preds)
    num_classes = len(unique_labels)
    
    # 计算每个类别的precision、recall、f1-score、support
    results_dict = {}
    results = ''

    # print(unique_labels)
    for labels in unique_labels:
        # print(label)
        current_label_result = []
        # TP + FN
        tp_fn = y_trues.count(labels)

        label0 = labels.split('+')[0]
        label1 = labels.split('+')[1]
        
        # TP
        tp = 0
        for i in range(len(y_trues)):
            if y_trues[i] == labels and label0 in y_preds[i]:
                tp += 1
            if y_trues[i] == labels and label1 in y_preds[i]:
                tp += 1

        support = tp_fn * 2
        # print(label, support, tp)

        try:
            precision = round(tp/support, digs)

        except ZeroDivisionError:
            precision = 0
            recall = 0
            f1_score = 0

        current_label_result.append(precision)
        current_label_result.append(support)
        # 输出第一行
        results_dict[str(labels)] = current_label_result
    title = '\t' + 'precision@' + str(k) + '\t' + 'num' + '\n'
    results += title

    for k, v in sorted(results_dict.items()):
        if (str(v[1]) == '0'):
            continue
        else:
            current_line = str(
                k) + '\t\t' + str(v[0]) + '\t\t' + str(v[1]) + '\n'
            results += current_line
            svc_pr1 += v[0]
            
    sums = len(y_trues)

    s_map = round(svc_pr1 / num_classes, digs)

    weighted_avg_results = [(v[0]*v[1]) for k, v in sorted(results_dict.items())]

    # 计算weighted avg
    weighted_precision = 0

    for weighted_avg_result in weighted_avg_results:
        weighted_precision += weighted_avg_result

    weighted_precision /= sums
    weighted_precision = round(weighted_precision / 2, digs)

    weighted_avg_line = 'avg:' + '\t\t' + str(round(weighted_precision, digs)) + '\t\t' + str(sums) + '\n' + 'map:' + '\t\t' + str(s_map)

    results += weighted_avg_line

    print(results)
    print()

    return weighted_precision, s_map

def getPredictions(pre_list):
    prediction = []
    for pre in pre_list:
        item = []
        results = pre.replace('[(', '').replace(')]', '').split('), (')
        for res in results:
            pre_svc = res.split(',')[0].replace('\'', '')
            item.append(pre_svc)
        prediction.append(item)
    return prediction


if __name__ == '__main__':
    timename = './results/pr1_map_stat.csv'
    stat_df = pd.DataFrame(columns=['type', 'ms', 'mRCA', 'tRCA'])                  
   
    for ben in f1_benchmarks:
        
        pr1_list = ['PR@1']
        map1_list = ['MAP@1']
        print('==== ' + ben + ' ====')
        for rca in rca_arr:
            print(rca)
            res_df = pd.read_csv(ben + rca + '_results.csv')
            new_col = ['time', 'fault', 'type', 'pred']
            res_df.columns = new_col
            
            test_k = 50
            y_true = res_df['fault'][:test_k].tolist()
            y_pred = getPredictions(res_df['pred'][:test_k].tolist())

            pr1, map1 = precision_k_f1(y_true, y_pred, k=topK, digs=2)

            pr1_list.append(pr1)
            map1_list.append(map1)

    stat_df.loc[len(stat_df)] = pr1_list
    stat_df.loc[len(stat_df)] = map1_list

    for ben in f2_benchmarks:
        
        pr1_list = ['PR@2']
        map1_list = ['MAP@2']
        print('==== ' + ben + ' ====')
        for rca in rca_arr:
            print(rca)
            res_df = pd.read_csv(ben + rca + '_results.csv')
            new_col = ['time', 'fault', 'type', 'pred']
            res_df.columns = new_col
            
            test_k = 50
            y_true = res_df['fault'][:test_k].tolist()
            y_pred = getPredictions(res_df['pred'][:test_k].tolist())

            pr1, map1 = precision_k_f2(y_true, y_pred, k=topK, digs=2)

            pr1_list.append(pr1)
            map1_list.append(map1)

    stat_df.loc[len(stat_df)] = pr1_list
    stat_df.loc[len(stat_df)] = map1_list
    
    print(stat_df)
    stat_df.to_csv('results/pr1_map1_stat.csv', index=None)