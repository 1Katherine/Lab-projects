import shutil
import time
import sys
import os
import json
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import joblib
import warnings
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from init_bayes_opt import JSONLogger as init_JSONLogger
from init_bayes_opt import Events as init_Events
from init_bayes_opt import BayesianOptimization as init_BayesianOptimization
from rs_bayes_opt import JSONLogger as rs_JSONLogger
from rs_bayes_opt import Events as rs_Events
from rs_bayes_opt import BayesianOptimization as rs_BayesianOptimization
import matplotlib.pyplot as plt
# # 调用代码：python feature_selection_bo.py --sampleType=all --ganrsGroup=4 --niters=10 --initFile=/usr/local/home/yyq/bo/ganrs_bo/wordcount-100G-GAN.csv
# import argparse
# parser = argparse.ArgumentParser(description='manual to this script')
# parser.add_argument('--benchmark', type=str, help='benchmark type')
# parser.add_argument('--niters', type=int, help='The number of iterations of the Bayesian optimization algorithm')
# parser.add_argument('--ninits', type=int, help='The number of initsamples of the Bayesian optimization algorithm')
# args = parser.parse_args()
# print('--niters = ' + str(args.niters) + '\t --ninits = ' + args.ninits)

# 获取当前文件路径
current_path = os.path.abspath(__file__)
# 获取当前文件的父目录,比如/usr/local/home/yyq/bo/rs_bo/rs_bo_newEI
father_path = os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".")
print(father_path)

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

def black_box_function(**params):
    print(params)
    i = []
    model = build_training_model(name)
    for conf in vital_params['vital_params']:
        i.append(params[conf])
    y = model.predict(np.matrix([i]))[0]
    return y


def draw_target(bo):
    # 画图
    plt.plot(-bo.space.target, label='bo  init_points = ' + str(ninits) + ', n_iter = ' + str(niters))
    max = bo._space.target.max()
    max_indx = bo._space.target.argmax()
    # 在图上描出执行时间最低点
    plt.scatter(max_indx, -max, s=20, color='r')
    plt.annotate('maxIndex:' + str(max_indx + 1), xy=(max_indx, -max), xycoords='data', xytext=(+20, +20),
                 textcoords='offset points'
                 , fontsize=12, arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad = .2'))
    plt.annotate(str(round(-max, 2)) + 's', xy=(max_indx, -max), xycoords='data', xytext=(+20, -20),
                 textcoords='offset points'
                 , fontsize=16, arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad = .2'))
    plt.xlabel('iterations')
    plt.ylabel('runtime')
    plt.legend()
    plt.savefig(father_path + "/target_"+ str(iterations) +".png")
    plt.show()

def rs_bo(ninit, iter, vital_params_range):
    # 定义贝叶斯优化模型
    optimizer = rs_BayesianOptimization(
        f=black_box_function,
        pbounds=vital_params_range,
        verbose=2,
        random_state=1,
    )
    logpath = father_path + "/logs_"+ str(iterations) +".json"
    logger = rs_JSONLogger(path=logpath)
    optimizer.subscribe(rs_Events.OPTIMIZATION_STEP, logger)

    optimizer.maximize(init_points=ninit, n_iter=iter, acq='ei')
    print(optimizer.max)
    draw_target(optimizer)

    vital_names = sorted(vital_params_range)  # sorted(vital_params_range) = vital_params_name
    res_df = getDF(logpath, vital_names)

    df_data = res_df[vital_names]
    df_target = res_df['target']
    from feature_selection.corraltion import feature_selected_K
    # 降维后的重要参数名称(减少10个参数）
    vitual_params = feature_selected_K(df_data, df_target, df_data.shape[1] - 1, vital_names)
    return res_df, vitual_params


def init_bo(initsamples, iter, vital_params_range):
    optimizer = init_BayesianOptimization(
            f=black_box_function,
            pbounds=vital_params_range,
            verbose=2,
            random_state=1,
            # bounds_transformer=bounds_transformer,
            custom_initsamples=initsamples
        )
    logpath = father_path + "/logs_"+ iterations +".json"
    logger = init_JSONLogger(path=logpath)
    optimizer.subscribe(init_Events.OPTIMIZATION_STEP, logger)

    init_points = len(initsamples)
    n_iter = iter
    print('interations：' + str(n_iter))
    optimizer.maximize(init_points=init_points, n_iter=n_iter, acq='ei')
    print(optimizer.max)
    draw_target(optimizer)

    vital_names = sorted(vital_params_range)  # sorted(vital_params_range) = vital_params_name
    res_df = getDF(logpath, vital_names)

    df_data = res_df[vital_names]
    df_target = res_df['target']
    from feature_selection.corraltion import feature_selected_K
    # 降维后的重要参数名称(减少10个参数）
    vitual_params = feature_selected_K(df_data, df_target, df_data.shape[1] - 1, vital_names)
    return res_df, vitual_params


def getDF(logpath, vital_names):
    # 存储所有样本数据
    # 读取json文件, 转成csv
    res_df = pd.DataFrame()
    for line in open(logpath).readlines():
        one_res = {}
        js_l = json.loads(line)
        one_res['target'] = -js_l['target']
        for pname in vital_names:
            one_res[pname] = js_l['params'][pname]
        df = pd.DataFrame(one_res, index=[0])
        res_df = res_df.append(df)
    # 设置索引从1开始
    res_df.index = range(1, len(res_df) + 1)
    res_df.to_csv(father_path + "/generationConf_"+ str(iterations) +".csv")
    return res_df

# --------------------- 读取初始样本 初始 start -------------------
# 取所有样本作为bo初始样本
def get_init_samples(initsamples_df, params_names):
    # 初始样本
    vital_params_list = params_names.append('runtime')
    initsamples = initsamples_df[vital_params_list].to_numpy()
    print('从csv文件中获取初始样本:' + str(len(initsamples)))
    return initsamples
# --------------------- 生成 gan-rs 初始 end -------------------

if __name__ == '__main__':
    name = 'xgb'
    # 重要参数
    vital_params_path = father_path +   '/files100/' + name + "/selected_parameters.txt"
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
    # 遍历训练数据中的参数，读取其对应的参数空间
    d1={}
    d2={}
    for conf in vital_params['vital_params']:
        if conf in confDict:
            d1 = {conf: (confDict[conf]['min'], confDict[conf]['max'])}
            d2.update(d1)
        else:
            print(conf,'-----参数没有维护: ', '-----')

    # 按照贝叶斯优化中的key顺序,得到重要参数的名称vital_params_name用于把json结果文件转成dataframe存成csv，以及重要参数+执行时间列vital_params_list用于读取初始样本
    print('获取初始样本时，按照贝叶斯内部的key顺序传初始样本和已有的执行时间：')
    vital_params_name = sorted(d2)
    print('vital_params_name = ' + str(vital_params_name))

    ninits = 2
    niters = 5

    iterations = 1
    vitual_params = []
    # res_samples_df = []
    max_niterations = niters
    current_niterations = 0
    # ------------------ 第一次迭代bo：使用随机生成样本 -------------
    while current_niterations + 5 <= max_niterations:
        if iterations == 1:
            res_samples_df, vitual_params = rs_bo(ninit = ninits, iter = 3, vital_params_range = d2)
            print('第 ' + str(iterations) + ' 次迭代的结果样本为\n' + str(res_samples_df))
            iterations += 1
            current_niterations += 3
        else:
            init_samples = get_init_samples(initsamples_df=res_samples_df, params_names=sorted(d2))
            print('第 ' + str(iterations) + ' 次迭代使用的初始样本为\n' + str(init_samples))
            if len(sorted(d2)) < 10:
                res_samples_df, vitual_params = init_bo(init_samples, iter=max_niterations - current_niterations, vital_params_range=d2)
                print('第 ' + str(iterations) + ' 次迭代的结果样本为\n' + str(res_samples_df))
            else :
                # 遍历训练数据中的参数，读取其对应的参数空间
                d1 = {}
                d2 = {}
                for conf in vitual_params:
                    if conf in confDict:
                        d1 = {conf: (confDict[conf]['min'], confDict[conf]['max'])}
                        d2.update(d1)
                    else:
                        print(conf, '-----参数没有维护: ', '-----')
                res_samples_df, vitual_params = init_bo(init_samples, iter = 5, vital_params_range = d2)
                iterations += 1
                current_niterations += 5


