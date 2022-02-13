#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：3 GA优化代码 
@File ：ganrs_test.py
@Author ：Yang
@Date ：2022/1/13 15:42 
'''
import numpy as np
import pandas as pd
# 读取重要参数的特征值
name = 'rf'
# 模型选出的重要参数
vital_params_path = './files44/' + name + "/selected_parameters.txt"
vital_params = pd.read_csv(vital_params_path)
print(vital_params)

# 重要参数的特征值
parameters_features_path = './files44/' + name + "/parameters_features.txt"
parameters_features_file = []
parameters_features = []
vital_params_list = []
for line in open(parameters_features_path, encoding='gb18030'):
    parameters_features_file.append(line.strip())
# 取出重要参数
parameters_features_file = parameters_features_file[-len(vital_params):]
for conf in vital_params['vital_params']:
    for para in parameters_features_file:
        if conf in para:
            # 重要参数列表
            vital_params_list.append(para.split(':')[0])
            # 重要参数的特征重要值列表
            parameters_features.append(float(para.split(':')[1]))

# 初始样本
initpoint_path = 'wordcount-100G-GAN-44.csv'
initsamples = pd.read_csv(initpoint_path)
samples = initsamples[vital_params_list].to_numpy()
print(samples)