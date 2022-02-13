import datetime
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
import lightgbm as lgb
import xgboost
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import joblib
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from sko.GA import GA
import warnings


warnings.filterwarnings("ignore")

# 主机上运行的代码

'''
    不重新建模，使用已经构建好的模型
'''
def build_training_model(name):
    import warnings
    if name.lower() == "lgb":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            model = joblib.load(modelfile + 'lgb/lgb.pkl')
    elif name.lower() == "gbdt":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            model = joblib.load(modelfile + 'gbdt/gbdt.pkl')
    elif name.lower() == "rf":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            model = joblib.load(modelfile + 'rf/rf.pkl')
    elif name.lower() == 'xgb':
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            model = joblib.load(modelfile + 'xgb/xgb.pkl')
    elif name.lower() == 'ada':
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            model = joblib.load(modelfile + 'ada/ada.pkl')
    else:
        raise Exception("[!] There is no option ")
    return model


'''
    黑盒模型，传入参数，计算target（根据模型预测参数的执行时间）
'''
def black_box_function(params):
    model = build_training_model(name)
    runtime = model.predict(np.matrix([params]))[0]
    return runtime

if  __name__ == '__main__':
    name = 'gbdt'
    modelfile = './random_terasort-20G_error_analysis_self/102/'
    sizePop = 30
    maxIter = 30
    # 重要参数
    vital_params_path = modelfile + name + "/selected_parameters.txt"
    # 维护的参数-范围表
    conf_range_table = "Spark_conf_range_terasort.xlsx"


    '''
        读取模型输出的重要参数
    '''
    vital_params = pd.read_csv(vital_params_path)
    # 参数范围和精度，从参数范围表里面获取

    # 参数范围表
    sparkConfRangeDf = pd.read_excel(conf_range_table)
    # SparkConf 列存放的是配置参数的名称
    sparkConfRangeDf.set_index('SparkConf', inplace=True)
    # 转化后的字典形式：{index(值): {column(列名): value(值)}}
    # {'spark.broadcast.blockSize': {'Range': '32-64m', 'min': 32.0, 'max': 64.0, 'pre': 1.0, 'unit': 'm'}
    confDict = sparkConfRangeDf.to_dict('index')

    '''
        获取格式
    '''
    # 遍历训练数据中的参数，读取其对应的参数空间
    confLb = []  # 参数空间上界
    confUb = []  # 参数空间下界
    precisions = []  # 参数精度
    vital_params_list = []  # 重要参数
    for conf in vital_params['vital_params']:
        if conf in confDict:
            print(conf + '\t 下界:' + str(confDict[conf]['min']) + '\t 上界:' + str(confDict[conf]['max']) + '\t 精度:' + str(confDict[conf]['pre']))
            vital_params_list.append(conf)
            confLb.append(confDict[conf]['min'])
            confUb.append(confDict[conf]['max'])
            precisions.append(confDict[conf]['pre'])
        else:
            print(conf, '-----参数没有维护: ', '-----')
    # ------------新增代码 start--------------
    # 重要参数的特征值
    parameters_features_path = modelfile  + name + "/parameters_features.txt"
    parameters_features_file = []
    parameters_features = []
    for line in open(parameters_features_path, encoding='gb18030'):
        parameters_features_file.append(line.strip())

    # 取出重要参数的特征值
    parameters_features_file = parameters_features_file[-len(vital_params):]
    print('\n重要参数和特征值列表 =  \n' + str(parameters_features_file))
    for conf in vital_params_list:
        for para in parameters_features_file:
            if conf in para:
                # 重要参数的特征重要值列表
                parameters_features.append(float(para.split(':')[1]))
    print('\n重要参数列表 = \n' + str(vital_params_list))
    print('\n重要参数的重要性值 = \n' + str(parameters_features))
    # ------------新增代码 end--------------
    '''
        开始遗传算法
    '''
    startTime = datetime.datetime.now()
    # 确定其他参数
    fitFunc = black_box_function  # 适应度函数
    nDim = len(vital_params)  # 参数个数
    # probMut = 0.01  # 变异概率
    for i,pf in enumerate(parameters_features):
        if pf == 0.0:
            parameters_features[i] = 0.01
    print('处理后的重要参数的重要性值 = \n' + str(parameters_features))
    probMut = np.array(parameters_features)

    ga = GA(func=fitFunc,  n_dim=nDim, size_pop=sizePop, max_iter=maxIter, prob_mut=probMut, lb=confLb, ub=confUb,
            precision=precisions)
    best_x, best_y = ga.run()
    endTime = datetime.datetime.now()
    searchDuration = (endTime - startTime).seconds
    startDay = str(startTime.strftime( '%Y-%m-%d'))
    generation_best_pic_directory = modelfile + 'result/binary_result/pic/' + startDay + '/'
    generation_best_directory = modelfile + 'result/binary_result/result/' + startDay + '/'
    if not os.path.exists(generation_best_pic_directory):
        os.makedirs(generation_best_pic_directory)
    if not os.path.exists(generation_best_directory):
        os.makedirs(generation_best_directory)
    generation_best_file = open(generation_best_directory + 'generation_best_' + modelfile.split('/')[1] + '.txt', 'a')
    print('\nbinary GA ' + name + ' ,sizePop=' + str(sizePop) + ' ,maxIter=' + str(maxIter), file=generation_best_file)
    print(vital_params_list, file=generation_best_file)
    print('best_x : ' + str(best_x), file=generation_best_file)
    print('best_y : ' + str(best_y), file=generation_best_file)
    generation_best = []
    print('generation_best \n', file=generation_best_file)
    for row, x in enumerate(ga.generation_best_X):
        temp = dict(zip(vital_params_list, x))
        temp['runtime'] = ga.generation_best_Y[row]
        generation_best.append(temp)
        print(str(temp), file=generation_best_file)
    df = pd.DataFrame(generation_best)
    df.to_csv(generation_best_directory + 'generation_best '+ name + ' sizePop=' + str(sizePop) + ' maxIter=' + str(maxIter) + ' ' + str(startTime.strftime( '%Y-%m-%d %H-%M-%S')) + '.csv', index=False)

    # %% Plot the binary_result
    import pandas as pd
    import matplotlib.pyplot as plt

    Y_history = pd.DataFrame(ga.all_history_Y)
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(Y_history.index, Y_history.values, '.', color='red')
    ax[0].set_title('binary GA ' + name)
    Y_history.min(axis=1).cummin().plot(kind='line')
    plt.savefig(generation_best_pic_directory + 'binary ' + name + ' sizePop=' + str(sizePop) + ' maxIter=' + str(maxIter) + ' ' + str(startTime.strftime( '%Y-%m-%d %H-%M-%S')) + '.png')
    plt.show()




