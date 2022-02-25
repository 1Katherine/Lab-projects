import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import time
import numpy as np
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import os
from os.path import join as pjoin
import json
import random
from bayes_scode import JSONLogger, Events, BayesianOptimization,SequentialDomainReductionTransformer
from bayes_scode.sgan import train
from bayes_scode.configuration import parser
from bayes_scode.get_samples import ganrs_samples_all

args = parser.parse_args()
print('benchmark = ' + args.benchmark + '\t gan+rs生成的样本数：initpoints = ' + str(args.initpoints) + '\t bo迭代搜索次数：--niters = ' + str(args.niters))

# 获取当前文件路径
current_path = os.path.abspath(__file__)
# 获取当前文件的父目录,比如/usr/local/home/yyq/bo/rs_bo/rs_bo_newEI
father_path = os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".")

# 服务器运行spark时config文件
config_run_path = father_path + "/config/" + args.benchmark +  "/"
# 重要参数
vital_params_path = father_path + "/parameters_set.txt"
# 维护的参数-范围表
conf_range_table = father_path + "/Spark_conf_range_"  + args.benchmark.split('-')[0] +  ".xlsx"
# 保存配置
generation_confs = father_path + "/generationConf.csv"

# --------------------- 生成 gan-rs 初始种群 start -------------------
# 产生n个随机初始样本给gan做训练
def random_for_gan(n):
    params_list=[]
    for param in vital_params_name:
          params_list.append(param)
    params_list.append('runtime')
    rs_df_all=pd.DataFrame(columns=params_list)
    for i in range(n):
        config=[]
        for conf in vital_params_name:
            if conf in confDict:
                min=confDict[conf]['min']
                max=confDict[conf]['max']
                pre=confDict[conf]['pre']
            else:
                raise Exception(conf, '-----参数没有维护: ', '-----')
            if pre==0.01:
                k=round(random.uniform(min,max),2)
            else:
                k=random.randint(min,max)
            config.append(k)
        print('随机生成的配置:' + str(config))
        y = schafferRun(config)
        config.append(y)
        print('随机生成的配置和实际运行时间:' + str(config))
        df_oneConfig_withRuntime = pd.DataFrame(data=[config],columns=params_list)
        print(df_oneConfig_withRuntime)
        rs_df_all=rs_df_all.append(df_oneConfig_withRuntime, ignore_index=True)
        print(rs_df_all)
    return rs_df_all


def gan_random(n, type, bo_res_df, iterations):
    print('gan_random方法获取的参数:' + '\t n = ' + str(n) + '\t type = ' + str(type) + '\t bo_res_df = ' + str(bo_res_df) + '\t iterations = ' + str(iterations))
    args = parser.parse_args()
    first_time = time.time()
    base = args.initNumber
    if n%base !=0:
        raise Exception("gan+random的采样方式为3+3，每轮的配置为6个!")
    params_list=[]
    for param in vital_params_name:
          params_list.append(param)
    params_list.append('runtime')
    #dataset存储所有的数据
    dataset=pd.DataFrame(columns=params_list)
    # 随机生成两个配置并放到服务器上运行
    if type == 'random':
        train_df_all = random_for_gan(base // 2)
        times = base // 2
    elif type == 'bo_result':
        train_df_all = bo_res_df[params_list]
        times = base // 2
    else:
        print('请在random和bo_result类型中二选一！')
    train_df_all = train_df_all.sort_values('runtime').reset_index(drop=True)
    # ------------------------- 第一轮gan：gan拿随机样本中最好的一个做训练，再从gan生成的9个假样本中取执行时间最短的两个样本 start ----------------------------
    bestconfig = train_df_all.iloc[:1,:-1]
    print(bestconfig)
    generate_data=train(bestconfig, first_time, args)
    print(generate_data)

    for i in range(times):
        # 取gan的前两个生成样本，实际运行
        config=generate_data.iloc[i].tolist()
        y = schafferRun(config)
        config.append(y)
        df_oneConfig_withRuntime = pd.DataFrame(data=[config], columns=params_list)
        print(df_oneConfig_withRuntime)
        train_df_all = train_df_all.append(df_oneConfig_withRuntime, ignore_index=True)
        print(train_df_all)
    # 训练样本 + gan生成的样本存入dataset
    dataset=dataset.append(train_df_all,ignore_index=True)
    print(dataset)
    dataset.to_csv('dataset_'+ str(iterations) +'.csv')
    return dataset


# --------------------- 生成 gan-rs 初始种群 end -------------------

def black_box_function(**params):
    i=[]
    for conf in vital_params_name:
        i.append(params[conf])
    print('black_box_function中schafferRun(i) 中的i为' + str(i))
    return -schafferRun(i)


# 格式化参数配置：精度、单位等
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


# 1、单个配置 p写入到 /usr/local/home/zwr/hibench-spark-config/wordcount-100G-ga   命名：config1
# 2、run获取执行时间并返回
last_runtime = 1.0
def run(configNum):
    # configNum = None
    # 使用给定配置运行spark
    run_cmd = '/usr/local/home/zwr/' + args.benchmark + '-ga.sh ' + str(
        configNum) + ' ' + father_path + '/config/' + args.benchmark
    print('configNum = ' + str(configNum) + '\t run_cmd = ' + str(run_cmd))
    if os.system(run_cmd) == 0:
        print('run_cmd命令执行成功')
    else:
        print('run_cmd命令执行失败')
    # 睡眠3秒，保证hibench.report文件完成更新后再读取运行时间
    time.sleep(3)
    # 获取此次spark程序的运行时间
    get_time_cmd = 'tail -n 1 /usr/local/home/hibench/hibench/report/hibench.report'
    line = os.popen(get_time_cmd).read()
    runtime = float(line.split()[4])
    global last_runtime
    if runtime == last_runtime:
        runtime = 100000.0
    else:
        last_runtime = runtime
    return runtime


# 1、实际运行
configNum = 1
def schafferRun(p):
    global configNum
    # 打开配置文件模板
    fTemp = open('configTemp', 'r')
    # 复制模板，并追加配置
    fNew = open(config_run_path + 'config' + str(configNum), 'a+')
    shutil.copyfileobj(fTemp, fNew, length=1024)
    try:
        for i in range(len(p)):
            fNew.write(' ')
            fNew.write(vital_params_name[i])
            fNew.write('\t')
            fNew.write(formatConf(vital_params_name[i], p[i]))
            fNew.write('\n')
    finally:
        fNew.close()
    runtime = run(configNum)
    configNum += 1
    return runtime


def draw_target(bo, iterations):
    # 画图
    plt.plot(-bo.space.target, label='ganrs_bo  init_points = ' + str(init_points) + ', n_iter = ' + str(n_iter))
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
    plt.close('all')

# 读取json文件, 转成csv
def bo_result_jsonTocsv(logpath, iterations):
    res_df = pd.DataFrame()
    for line in open(logpath).readlines():
        one_res = {}
        js_l = json.loads(line)
        one_res['runtime'] = -js_l['runtime']
        for pname in vital_params_name:
            one_res[pname] = js_l['params'][pname]
        df = pd.DataFrame(one_res, index=[0])
        res_df = res_df.append(df)
    # 设置索引从1开始
    res_df.index = range(1, len(res_df)+1)
    res_df.to_csv(generation_confs + str(iterations) + ".csv")
    return res_df

if __name__ == '__main__':
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
    precisions = []  # 参数精度
    for conf in vital_params['vital_params']:
        if conf in confDict:
            d1 = {conf: (confDict[conf]['min'], confDict[conf]['max'])}
            d2.update(d1)
            precisions.append(confDict[conf]['pre'])
        else:
            print(conf,'-----参数没有维护: ', '-----')

    # d2按照key值排序
    print('按照key值排序前的d2 = ' + str(d2))
    sort_dict = {}
    for k in sorted(d2):
        sort_dict[k] = d2[k]
    d2 = sort_dict
    print('按照key值排序后的d2 = ' + str(d2))

    vital_params_name = sorted(d2)
    print('vital_params_name = ' + str(vital_params_name))
    vital_params_list = sorted(d2)
    vital_params_list.append('runtime')
    print('vital_params_list = ' + str(vital_params_list))

    iterations = 0
    # 记录贝叶斯过程中使用的初始样本和探索样本
    res_df = pd.DataFrame()
    maxIters = 3
    while iterations < maxIters:
        if iterations == 0:
            dataset = gan_random(n=6, type='random', bo_res_df=res_df, iterations=iterations)
            initsamples = ganrs_samples_all(initsamples_df=dataset)

            #定义贝叶斯优化模型
            optimizer = BayesianOptimization(
                    f=black_box_function,
                    pbounds=d2,
                    verbose=2,
                    random_state=1,
                    custom_initsamples=initsamples
                )
            logpath = father_path + "logs_"+ str(iterations) +".json"
            logger = JSONLogger(path=logpath)
            optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

            init_points = len(initsamples)
            n_iter = args.niters
            optimizer.maximize(init_points=init_points, n_iter=n_iter, acq='ei')
            print(optimizer.max)
            draw_target(optimizer, iterations)
            res_df = bo_result_jsonTocsv(logpath)
            iterations += 1
        else :
            print('iterations = ' + str(iterations))

            dataset = gan_random(n=6, type='bo_result', bo_res_df=res_df, iterations=iterations)

            initsamples = ganrs_samples_all(initsamples_df=dataset)
            #定义贝叶斯优化模型
            optimizer = BayesianOptimization(
                    f=black_box_function,
                    pbounds=d2,
                    verbose=2,
                    random_state=1,
                    custom_initsamples=initsamples
                )
            logpath = father_path + "logs_"+ str(iterations) +".json"
            logger = JSONLogger(path=logpath)
            optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

            init_points = len(initsamples)
            n_iter = args.niters
            optimizer.maximize(init_points=init_points, n_iter=n_iter, acq='ei')
            print(optimizer.max)
            draw_target(optimizer, iterations)
            res_df = bo_result_jsonTocsv(logpath)
            iterations += 1
