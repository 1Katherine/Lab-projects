import pandas as pd
# 导入后加入以下列，再显示时显示完全。
pd.set_option('display.max_rows',500)
pd.set_option('display.max_columns',500)
pd.set_option('display.width',1000)
import numpy as np
import warnings
from sklearn.ensemble import GradientBoostingRegressor
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
import joblib
import sys
import os
import json
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from init_bayes_opt import JSONLogger as init_JSONLogger
from init_bayes_opt import Events as init_Events
from init_bayes_opt import BayesianOptimization as init_BayesianOptimization
from rs_bayes_opt import JSONLogger as rs_JSONLogger
from rs_bayes_opt import Events as rs_Events
from rs_bayes_opt import BayesianOptimization as rs_BayesianOptimization
import matplotlib.pyplot as plt
# 调用代码：python feature_selection_bo_laptop.py --sampleType=all --ganrsGroup=4 --niters=10 --initFile=/usr/local/home/yyq/bo/ganrs_bo/wordcount-100G-GAN.csv
import argparse
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--benchmark', type=str, default='wordcount-100G', help='benchmark type')
parser.add_argument('--niters', type=int, default=20, help='The number of iterations of the Bayesian optimization algorithm')
parser.add_argument('--ninits', type=int, default=2, help='The number of initsamples of the Bayesian optimization algorithm')
args = parser.parse_args()
print('--niters = ' + str(args.niters) + '\t --ninits = ' + str(args.ninits))

# 获取当前文件路径
current_path = os.path.abspath(__file__)
# 获取当前文件的父目录,比如/usr/local/home/yyq/bo/rs_bo/rs_bo_newEI
father_path = os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".")

# 获取当前文件路径
current_path = os.path.abspath(__file__)
# 获取当前文件的父目录
father_path = os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".")

# 服务器运行spark时config文件
config_run_path = father_path + "/config/" + args.benchmark +  "/"
# 重要参数
vital_params_path = father_path + "/parameters_set.txt"
# 维护的参数-范围表
conf_range_table = father_path + "/Spark_conf_range_"  + args.benchmark.split('-')[0] +  ".xlsx"
# 保存配置
# generation_confs = father_path + "/generationConf.csv"


warnings.filterwarnings("ignore")

'''
    根据名称构建模型
'''
def build_model(name):
    if name.lower() == "lgb":
        model = lgb.LGBMRegressor()
    elif name.lower() == "gbdt":
        model = GradientBoostingRegressor()
    else:
        model = RandomForestRegressor()
    return model


'''
    不重新建模，使用已经构建好的模型
'''
def build_training_model(name):
    if name.lower() == "lgb":
        model = joblib.load(modelfile + 'lgb/lgb.pkl')
    elif name.lower() == "gbdt":
        model = joblib.load(modelfile + 'gbdt/gbdt.pkl')
    elif name.lower() == "rf":
        model = joblib.load(modelfile + 'rf/rf.pkl')
    elif name.lower() == 'xgb':
        model = joblib.load(modelfile + 'xgb/xgb.pkl')
    elif name.lower() == 'ada':
        model = joblib.load(modelfile + 'ada/ada.pkl')
    else:
        raise Exception("[!] There is no option ")
    return model


'''
    贝叶斯的黑盒模型，传入参数，计算target（根据模型预测参数的执行时间）
'''
def black_box_function(**params):
    i = []
    model = build_training_model(name)
    for conf in vital_params['vital_params']:
        i.append(params[conf])
    y = model.predict(np.matrix([i]))[0]
    return -y

def draw_target(bo):
    # 画图
    plt.plot(-bo.space.target, label='bo  init_points = ' + str(args.ninits) + ', n_iter = ' + str(args.niters))
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
        # bounds_transformer=bounds_transformer
    )
    logpath = father_path + "/logs_"+ str(iterations) +".json"
    logger = rs_JSONLogger(path=logpath)
    optimizer.subscribe(rs_Events.OPTIMIZATION_STEP, logger)
    optimizer.maximize(init_points=ninit, n_iter=iter, acq='ei')

    # 存储所有样本数据
    # 读取json文件, 转成csv
    res_df = pd.DataFrame()
    for line in open(logpath).readlines():
        one_res = {}
        js_l = json.loads(line)
        one_res['runtime'] = -js_l['target']
        vital_names = sorted(vital_params_range)  # sorted(vital_params_range) = vital_params_name
        for pname in vital_names:
            one_res[pname] = js_l['params'][pname]
        df = pd.DataFrame(one_res, index=[0])
        res_df = res_df.append(df)
    # 设置索引从1开始
    res_df.index = range(1, len(res_df) + 1)
    res_df.to_csv(father_path + "/generationConf_"+ str(iterations) +".csv")

    df_data = res_df[vital_names]
    df_target = res_df['runtime']
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
    logpath = father_path + "/logs_"+ str(iterations) +".json"
    logger = init_JSONLogger(path=logpath)
    optimizer.subscribe(init_Events.OPTIMIZATION_STEP, logger)

    init_points = len(initsamples)
    n_iter = iter
    print('interations：' + str(n_iter))
    optimizer.maximize(init_points=init_points, n_iter=n_iter, acq='ei')
    print(optimizer.max)
    draw_target(optimizer)


    #存储数据
    # 读取json文件, 转成csv
    import json
    res_df = pd.DataFrame()
    for line in open(logpath).readlines():
        one_res = {}
        js_l = json.loads(line)
        one_res['runtime'] = -js_l['target']
        vital_names = sorted(vital_params_range)  # sorted(vital_params_range) = vital_params_name
        for pname in vital_names:
            one_res[pname] = js_l['params'][pname]
        df = pd.DataFrame(one_res, index=[0])
        res_df = res_df.append(df)
    # 设置索引从1开始
    res_df.index = range(1, len(res_df)+1)
    res_df.to_csv(father_path + "/generationConf_" + str(iterations) + ".csv")

    df_data = res_df[vital_names]
    df_target = res_df['runtime']
    from feature_selection.corraltion import feature_selected_K
    # 降维后的重要参数名称(减少10个参数）
    vitual_params = feature_selected_K(df_data, df_target, df_data.shape[1] - 10, vital_names)
    return res_df, vitual_params

# --------------------- 读取初始样本 初始 start -------------------
# 取所有样本作为bo初始样本
def get_init_samples(initsamples_df, params_names):
    # 初始样本
    params_names.append('runtime')
    vital_params_list = params_names
    print('res_samples_df = \n' + str(initsamples_df))
    print('res_samples_df.columns = \n' + str(initsamples_df.columns) +'res_samples_df.columns.len = \n' + str(len(initsamples_df.columns)) +
          '\nvital_params_names = \n' + str(params_names) + '\nvital_params_names.len = \n' + str(len(params_names)))
    # print(sorted(vital_params_list) == sorted(initsamples_df.columns))
    print('vital_params_list = ' + str(vital_params_list) + ', type = ' + str(type(vital_params_list)))
    for col in vital_params_list:
        print(col)
        if col in initsamples_df.columns:
            print(col)
            print(initsamples_df[col])
        else:
            print(col + '在vital_params_list中，不在initsamples_df.columns中')
    print('initsamples_df[vital_params_list] = \n')
    print(initsamples_df[vital_params_list])
    initsamples = initsamples_df[vital_params_list].to_numpy()
    print('从csv文件中获取初始样本:' + str(len(initsamples)))
    return initsamples
# --------------------- 生成 gan-rs 初始 end -------------------

if __name__ == '__main__':
    name = 'rf'
    modelfile = './files30/'
    # 重要参数
    vital_params_path = modelfile + name + "/selected_parameters.txt"
    # 维护的参数-范围表
    conf_range_table = father_path + "\\Spark_conf_range_wordcount.xlsx"
    # 读取重要参数
    vital_params = pd.read_csv(vital_params_path)
    print('重要参数列表（将贝叶斯的x_probe按照重要参数列表顺序转成配置文件实际运行:')
    print(vital_params)
    # 参数范围和精度，从参数范围表里面获取
    sparkConfRangeDf = pd.read_excel(conf_range_table)
    sparkConfRangeDf.set_index('SparkConf', inplace=True)
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

    iterations = 1
    vitual_params = []
    res_samples_df = pd.DataFrame()
    max_niterations = args.niters
    current_niterations = 0

    # ------------------ 第一次迭代bo：使用随机生成样本 -------------
    while current_niterations + 5 <= max_niterations:
        print('进入current_niterations')
        if iterations == 1:
            res_samples_df, vitual_params = rs_bo(ninit = args.ninits, iter = 1, vital_params_range = d2)
            print('第 ' + str(iterations) + ' 次迭代的结果样本为\n' + str(res_samples_df))
            iterations += 1
            current_niterations += 1
        else:
            if len(sorted(d2)) < 1:
                vital_params = sorted(d2)
                res_samples_df, vitual_params = init_bo(init_samples, iter=max_niterations - current_niterations, vital_params_range=d2)
                print('第 ' + str(iterations) + ' 次迭代的结果样本为\n' + str(res_samples_df))
                break
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
                print('降维后的vital_params_range = ' + str(vitual_params))
                print('降维后的初始样本 = \n' + str(res_samples_df))
                vital_params = pd.DataFrame(sorted(vitual_params), columns=['vital_params'])
                init_samples = get_init_samples(res_samples_df, sorted(vitual_params))
                res_samples_df, vitual_params = init_bo(init_samples, iter = 5, vital_params_range = d2)
                iterations += 1
                current_niterations += 5


