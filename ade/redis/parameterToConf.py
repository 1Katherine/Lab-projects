#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：redis 
@File ：parameterToConf.py
@Author ：Yang
@Date ：2022/3/7 9:43 
'''
import os
import shutil
import pandas as pd
import random
import numpy as np

# 获取当前文件路径
current_path = os.path.abspath(__file__)
# 获取当前文件的父目录,比如/usr/local/home/yyq/bo/rs_bo/rs_bo_newEI
father_path = os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".")

benchmark = 'redis'

# 服务器运行spark时config文件
config_run_path = father_path + "/config/" + benchmark +  "/"
# 重要参数
vital_params_path = father_path + "/parameters_set_redis.txt"
# 维护的参数-范围表
conf_range_table = father_path + "/Spark_conf_range_"  + benchmark.split('-')[0] +  ".xlsx"
# 保存配置
generation_confs = father_path +  "/generationConf.csv"


def formatConf(conf, value):
    print('需要通过formatConf处理的数据 : conf = ' + str(conf) + '\t value = ' + str(value))
    res = ''
    # 处理精度
    if confDict[conf]['pre'] == 1:
        res = round(value)
    elif confDict[conf]['pre'] == 0.01:
        res = round(value, 2)
    # 添加单位
    if not pd.isna(confDict[conf]['unit']):
        # 布尔值
        if confDict[conf]['unit'] == 'flag':
            res = str(bool(res)).lower()
        # 列表形式的参数（spark.serializer、spark.io.compression.codec等）
        elif confDict[conf]['unit'] == 'list':
            rangeList = confDict[conf]['Range'].split(' ')
            res = rangeList[int(res)]
        # 拼接上单位
        else:
            res = str(res) + confDict[conf]['unit']
    else:
        res = str(res)
    return res

# 1、实际运行
configNum = 1
def schafferRun(p):
    global configNum
    # 打开配置文件模板
    fTemp = open('configTemp_redis', 'r')
    # 复制模板，并追加配置
    fNew = open(config_run_path + 'config' + str(configNum), 'a+')
    shutil.copyfileobj(fTemp, fNew, length=1024)
    try:
        for i in range(len(p)):
            if vital_params_name[i] == 'save1':
                fNew.write('  ')
                fNew.write('save')
                fNew.write('\t')
                fNew.write(formatConf(vital_params_name[i], p[i]))
                save2_index = vital_params_name.index("save2")
                fNew.write(' ')
                fNew.write(formatConf(vital_params_name[save2_index], p[save2_index]))
                fNew.write('\n')
            elif vital_params_name[i] == 'save2':
                pass
            else:
                fNew.write('  ')
                fNew.write(vital_params_name[i])
                fNew.write('\t')
                fNew.write(formatConf(vital_params_name[i], p[i]))
                fNew.write('\n')
    finally:
        fNew.close()
    runtime = run(configNum)
    configNum += 1
    return runtime

def run(configNum):
    print(configNum)

if __name__ == '__main__':
    # 读取重要参数
    vital_params = pd.read_csv(vital_params_path)
    print('重要参数列表（将贝叶斯的x_probe按照重要参数列表顺序转成配置文件实际运行:')
    print(vital_params)
    # 参数范围和精度，从参数范围表里面获取
    sparkConfRangeDf = pd.read_excel(conf_range_table)
    sparkConfRangeDf.set_index('SparkConf', inplace=True)
    confDict = sparkConfRangeDf.to_dict('index')

    d1={}
    d2={}
    for conf in vital_params['vital_params']:
        if conf in confDict:
            d1 = {conf: (confDict[conf]['min'], confDict[conf]['max'])}
            d2.update(d1)
        else:
            if conf == 'save':
                # save1
                conf1 = conf + '1'
                d1 = {conf1: (confDict[conf1]['min'], confDict[conf1]['max'])}
                d2.update(d1)
                # save2
                conf2 = conf + '2'
                d1 = {conf2: (confDict[conf2]['min'], confDict[conf2]['max'])}
                d2.update(d1)
            else:
                print(conf,'-----参数没有维护: ', '-----')

    vital_params_name = sorted(d2)
    print(vital_params_name)

    sort_dict = {}
    for k in sorted(d2):
        sort_dict[k] = d2[k]
    d2 = sort_dict
    print('按照key值排序后的d2 = ' + str(d2))


    print(d2)
    print(len(d2))
    i = []
    params = {}
    for kv in d2.items():
        # print(kv[1][0])
        params[kv[0]] = random.uniform(kv[1][0], kv[1][1])
    print(params)
    for conf in vital_params_name:
        i.append(params[conf])
    print(i)

    schafferRun(i)