#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：feature_selection_bo 
@File ：randomtree.py
@Author ：Yang
@Date ：2022/3/1 16:06 
'''
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import csv
import pandas as pd
import os
import numpy as np

# 获取当前文件路径
current_path = os.path.abspath(__file__)
# 获取当前文件的父目录
father_path = os.path.abspath(os.path.dirname(os.path.dirname(current_path)) + os.path.sep + '.')

params_names = []
def csvTodf():
    global params_names
    # ----------------------- 数据处理：csv转换成df ---------------------
    tmp_lst = []
    with open(father_path + '/generationConf/generationConf_1.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            tmp_lst.append(row)
    df = pd.DataFrame(tmp_lst[1:], columns=tmp_lst[0])
    # print(df)
    params_names = df.columns.tolist()[2:]
    df_data = df[params_names]
    df_target = df['runtime']
    df_data = df_data[:10]
    df_target = df_target[:10]
    print('df_data = \n' + str(df_data))
    print('target = \n' + str(df_target))
    return df_data, df_target


def feature_selected_K(df_data, df_target, k):
    x_train, x_test, y_train, y_test = train_test_split(df_data, df_target, test_size=0.3, random_state=0)
    feat_labels = df_data.columns
    forest = RandomForestRegressor(n_estimators=10000, random_state=0, n_jobs=-1)
    forest.fit(x_train, y_train)
    print('feat_labels : ' + str(feat_labels))
    importances = forest.feature_importances_
    indices = np.argsort(importances)[::-1]
    print('随机森林输出的重要参数和特征值')
    dict_corr = {}
    for f in range(x_train.shape[1]):
        print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))
        dict_corr[feat_labels[indices[f]]] = importances[indices[f]]
    d_order = sorted(dict_corr.items(), key=lambda x: x[1], reverse=True)
    # 获取score得分最高的前n个重要参数
    keys = []
    d_firstK = d_order[:k]
    for item in d_firstK:
        keys.append(item[0])
    print('前 ' + str(k) + ' 个重要参数名称 = ' + str(keys))
    # print(df_data[keys])
    return keys


if __name__ == '__main__':
    k = 1
    df_data, df_target = csvTodf()
    vitual_params = feature_selected_K(df_data, df_target, df_data.shape[1] - k)
    print(vitual_params)