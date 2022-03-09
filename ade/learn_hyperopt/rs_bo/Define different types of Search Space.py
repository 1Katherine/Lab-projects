#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：learn_hyperopt 
@File ：Define different types of Search Space.py
@Author ：Yang
@Date ：2022/2/28 16:36 
'''
# Import HyperOpt Library
from hyperopt import tpe, hp, fmin, Trials
import pandas as pd
from hyperopt.graph_viz import dot_hyperparameters
import numpy as np

import os
# 获取当前文件路径
current_path = os.path.abspath(__file__)
# 获取当前文件的父目录,比如/usr/local/home/yyq/bo/rs_bo/rs_bo_newEI
father_path = os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".")

name = 'rf'
# 重要参数
vital_params_path = './files100/' + name + "/selected_parameters.txt"
print(vital_params_path)
# 维护的参数-范围表
conf_range_table = "Spark_conf_range_wordcount.xlsx"
# 参数配置表（模型选出的最好配置参数）
generation_confs = father_path + "/generationConf.csv"

'''
    读取模型输出的重要参数
'''
vital_params = pd.read_csv(vital_params_path)
print(vital_params)
# 参数范围和精度，从参数范围表里面获取

# 参数范围表
sparkConfRangeDf = pd.read_excel(conf_range_table)
# SparkConf 列存放的是配置参数的名称
sparkConfRangeDf.set_index('SparkConf', inplace=True)
# 转化后的字典形式：{index(值): {column(列名): value(值)}}
# {'spark.broadcast.blockSize': {'Range': '32-64m', 'min': 32.0, 'max': 64.0, 'pre': 1.0, 'unit': 'm'}
confDict = sparkConfRangeDf.to_dict('index')

for conf in vital_params['vital_params']:
    if conf in confDict:
        print(conf, confDict[conf]['min'], confDict[conf]['max'], confDict[conf]['pre'])

import matplotlib.pyplot as plt

def draw_target(loss):
    np_loss = np.array(loss)
    # 画图
    plt.plot(loss, label='hyperopt max_evals = ' + str(max_evals))
    min_loss = np_loss.min()
    min_indx = np_loss.argmin()
    # 在图上描出执行时间最低点
    plt.scatter(min_indx, min_loss, s=20, color='r')
    plt.annotate('maxIndex:' + str(min_indx + 1), xy=(min_indx, min_loss), xycoords='data', xytext=(+20, +20),
                 textcoords='offset points'
                 , fontsize=12, arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad = .2'))
    plt.annotate(str(round(min_loss, 2)) + 's', xy=(min_indx, min_loss), xycoords='data', xytext=(+20, -20),
                 textcoords='offset points'
                 , fontsize=16, arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad = .2'))
    plt.xlabel('iterations')
    plt.ylabel('runtime')
    plt.legend()
    plt.savefig(father_path + "/target.png")
    plt.show()
    plt.close('all')

def f(params):
    # print('params\n' + str(params.keys))
    sum = 0
    for key in params:
        sum += params[key]
    # x1, x2 = params['x1'], params['x2']
    # return x1 * x2;
    return sum

search_space = {
    'x1': hp.uniform('x1', 0.5, 0.9),
    'x2': hp.randint('x2', int(1.0), int(4.0)),
    'x3': hp.randint('x3', int(3.0), int(7.0)),
    'x4': hp.randint('x4', int(4.0), int(8.0)),
    'x5': hp.uniform('x5', 0.5, 0.9),
    'x6': hp.randint('x6', int(16.0) ,int(48.0)),
    # 'x7': hp.randint('x7', int(1073741823.0) ,int(2147483647.0)),
    # 'x8': hp.randint('x8', 200.0, 500.0),
    # 'x9': hp.randint('x9', 24.0, 72.0),
    # 'x10': hp.randint('x10', 384.0, 877.0),
}
max_evals = 100
trials = Trials()
best = fmin(
    fn=f,
    space=search_space,
    algo=tpe.suggest,
    max_evals=max_evals,
    trials=trials
)

open('foo.dot', 'w').write(dot_hyperparameters(search_space))
print('best\n' + str(best))

# print(trials.results[0])
# print(trials.vals)


result = []
loss = []
for i in range(max_evals):
    res = {}
    for key in search_space:
        # print('trials.vals[key] : ' + str(key) + '   -   ' + str(trials.vals[key][i]))
        res[key] = trials.vals[key][i]
    res['loss'] = trials.results[i]
    loss.append(trials.results[i]['loss'])
    print(res)
#     i += 1
    result.append(res)
print('result\n' + str(result))
print(loss)

logpath = "output.txt"
f = open(logpath, 'w+')
for i, r in enumerate(result):
    print(r, file=f)
f.close()

draw_target(loss)

# 存储数据
# 读取json文件, 转成csv
import json
res_df = pd.DataFrame()
for line in open(logpath).readlines():
    one_res = {}
    line = line.replace("'", '"')
    js_l = json.loads(line)
    print(js_l)
    one_res['loss'] = js_l['loss']['loss']
    for pname in search_space:
        one_res[pname] = js_l[pname]
    df = pd.DataFrame(one_res, index=[0])
    res_df = res_df.append(df)
# 设置索引从1开始
res_df.index = range(1, len(res_df) + 1)
res_df.to_csv(generation_confs)