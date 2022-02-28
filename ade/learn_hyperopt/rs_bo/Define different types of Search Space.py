#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：learn_hyperopt 
@File ：Define different types of Search Space.py
@Author ：Yang
@Date ：2022/2/28 16:36 
'''
# Import HyperOpt Library
from hyperopt import tpe, hp, fmin
import pandas as pd

name = 'rf'
# 重要参数
vital_params_path = './files100/' + name + "/selected_parameters.txt"
print(vital_params_path)
# 维护的参数-范围表
conf_range_table = "Spark_conf_range_wordcount.xlsx"
# 参数配置表（模型选出的最好配置参数）
generation_confs = './searching_config/' + name + "generationbestConf.csv"

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


best = fmin(
    fn=f,
    space=search_space,
    algo=tpe.suggest,
    max_evals=100
)

print(best)