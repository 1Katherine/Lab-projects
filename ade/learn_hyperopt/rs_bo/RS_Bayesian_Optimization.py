import datetime
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import joblib
from bayes_opt import SequentialDomainReductionTransformer
from bayes_opt import BayesianOptimization
import warnings
# Import HyperOpt Library
from hyperopt import tpe, hp, fmin

warnings.filterwarnings("ignore")

'''
    根据名称构建模型
'''
def build_model(name):
    if name.lower() == "lgb":
        model = lgb.LGBMRegressor()
    elif name.lower() == "gdbt":
        model = GradientBoostingRegressor()
    else:
        model = RandomForestRegressor()
    return model


'''
    不重新建模，使用已经构建好的模型
'''
def build_training_model(name):
    if name.lower() == "lgb":
        model = joblib.load('./files100/lgb/lgb.pkl')
    elif name.lower() == "gbdt":
        model = joblib.load('./files100/gbdt/gbdt.pkl')
    elif name.lower() == "rf":
        model = joblib.load('./files100/rf/rf.pkl')
    elif name.lower() == 'xgb':
        model = joblib.load('./files100/xgb/xgb.pkl')
    elif name.lower() == 'ada':
        model = joblib.load('./files100/ada/ada.pkl')
    else:
        raise Exception("[!] There is no option ")
    return model


'''
    贝叶斯的黑盒模型，传入参数，计算target（根据模型预测参数的执行时间）
'''
# def black_box_function(**params):
#     i = []
#     model = build_training_model(name)
#     for conf in vital_params['vital_params']:
#         i.append(params[conf])
#     y = model.predict(np.matrix([i]))[0]
#     return y

def objective(params):
    print(params)
    i = []
    model = build_training_model(name)
    for conf in vital_params['vital_params']:
        i.append(params[conf])
    y = model.predict(np.matrix([i]))[0]
    return y


if __name__ == '__main__':
    name = 'xgb'
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

    '''
        获取pbounds格式 pbounds = {'x': (-5, 5), 'y': (-2, 15)}
    '''
    # 遍历训练数据中的参数，读取其对应的参数空间
    d1 = {}
    space = {}
    for conf in vital_params['vital_params']:
        if conf in confDict:
            if confDict[conf]['pre'] == 1.0:
                d1 = {conf: hp.randint(conf, int(confDict[conf]['min']), int(confDict[conf]['max']))}
            if confDict[conf]['pre'] == 0.01:
                d1 = {conf: hp.uniform(conf, confDict[conf]['min'], confDict[conf]['max'])}
            # d1 = {conf: hp.uniform(conf, confDict[conf]['min'], confDict[conf]['max'])}
            # 用 d1 字典更新 d2 字典（防止参数范围表中有重名的配置参数行，被多次添加到字典中）
            space.update(d1)
        else:
            print(conf, '-----参数没有维护: ', '-----')

    for conf in vital_params['vital_params']:
        print('search_space ' + str(space[conf]))

    '''
        开始贝叶斯优化，传入space = {
                            'x': hp.uniform('x', -6, 6),
                            'y': hp.uniform('y', -6, 6)
                        }
    '''
    print('space:' + str(space))
    best = fmin(
        fn=objective,  # Objective Function to optimize
        space=space,  # Hyperparameter's Search Space
        algo=tpe.suggest,  # Optimization algorithm (representative TPE)
        max_evals=1000  # Number of optimization attempts
    )
    print(best)
