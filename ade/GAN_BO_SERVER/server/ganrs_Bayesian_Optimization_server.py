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
generation_confs = "generationConf.csv"

# --------------------- 生成 gan-rs 初始种群 start -------------------
'''
2022/2/15
添加gan网络
'''
def gan_random(n):
    first_time = time.time()
    base = 6
    if n%base != 0:
        raise Exception("gan+random的采样方式为"+ str(base//2) + "+" + str(base//2) +"，每轮的配置为" + str(base) +"个，所以初始的样本点必须为" + str(base) + "的倍数！")
    #执行的轮数
    number=int(n/base)
    params_list=[]
    for param in vital_params['vital_params']:
          params_list.append(param)
    params_list.append('runtime')
    #dataset存储所有的数据
    dataset=pd.DataFrame(columns=params_list)
    m=pd.DataFrame(columns=params_list)
    for i in range(base//2):
        config=[]
        for conf in vital_params['vital_params']:
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

        print('np.matrix([config])中的config:\n' + str(config))
        # y=model.predict(np.matrix([config]))[0]
        print('np.matrix([config]):\n' + str(np.array(config)))
        y = schafferRun(config)
        print('y = np.matrix([config])[0]中的 y = ' + str(y))
        config.append(y)
        print(config)
        n=pd.DataFrame(data=[config],columns=params_list)
        print(n)
        m=m.append(n, ignore_index=True)
        print(m)
    m=m.sort_values('runtime').reset_index(drop=True)
    bestconfig=m.iloc[:1,:-1]
    print(bestconfig)
    generate_data=train(bestconfig, first_time, args)
    print(generate_data)
    for i in range(base//2):
        config=generate_data.iloc[i].tolist()
        # y = model.predict(np.matrix([config]))[0]
        y = schafferRun(config)
        config.append(y)
        n = pd.DataFrame(data=[config], columns=params_list)
        print(n)
        m = m.append(n, ignore_index=True)
        print(m)
    dataset=dataset.append(m,ignore_index=True)
    #已经进行了一轮对number-1
    number=number-1
    for i in range(number):
        #先根据上一轮4个配置中的最优配置生成样本
        m = m.sort_values('runtime').reset_index(drop=True)
        m1=pd.DataFrame(columns=params_list)
        bestconfig = m.iloc[:1, :-1]
        print(bestconfig)
        generate_data = train(bestconfig, first_time, args)
        print(generate_data)
        for i in range(base//2):
            config = generate_data.iloc[i].tolist()
            # y = model.predict(np.matrix([config]))[0]
            y = schafferRun(config)
            config.append(y)
            n = pd.DataFrame(data=[config], columns=params_list)
            m1 = m1.append(n, ignore_index=True)
        #随机采样生成两个配置
        for i in range(base//2):
            config = []
            for conf in vital_params['vital_params']:
                if conf in confDict:
                    min = confDict[conf]['min']
                    max = confDict[conf]['max']
                    pre = confDict[conf]['pre']
                else:
                    raise Exception(conf, '-----参数没有维护: ', '-----')
                if pre == 0.01:
                    k = round(random.uniform(min, max), 2)
                else:
                    k = random.randint(min, max)
                config.append(k)
            print(config)
            # y = model.predict(np.matrix([config]))[0]
            y = schafferRun(config)
            config.append(y)
            print(config)
            n = pd.DataFrame(data=[config], columns=params_list)
            m1 = m1.append(n, ignore_index=True)
        m=pd.DataFrame(m1,copy=True)
        dataset = dataset.append(m1, ignore_index=True)
    print(dataset)
    dataset.to_csv(father_path + '/dataset.csv')
    return dataset



# --------------------- 生成 gan-rs 初始种群 end -------------------

def black_box_function(**params):
    i=[]
    for conf in vital_params['vital_params']:
        i.append(params[conf])
    print('black_box_function中schafferRun(i) 中的i为' + str(i))
    return -schafferRun(i)


# 格式化参数配置：精度、单位等
def formatConf(conf, value):
    print('需要通过formatConf处理的数据 : conf = ' + str(conf) + '\t value = ' + str(value))
    res = ''
    # 处理精度
    if confDict[conf]['pre'] == 1:
        res = np.round(value)
    elif confDict[conf]['pre'] == 0.01:
        res = np.round(value, 2)
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
    # run_cmd = '/usr/local/home/zwr/wordcount-100G-ga.sh ' + str(configNum) + ' ' + father_path + '/config/wordcount-100G'
    run_cmd = '/usr/local/home/zwr/' + args.benchmark + '-ga.sh ' + str(
        configNum) + ' ' + father_path + '/config/' + args.benchmark
    print('run_cmd = ' + str(run_cmd))
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
            fNew.write(vital_params['vital_params'][i])
            fNew.write('\t')
            fNew.write(formatConf(vital_params['vital_params'][i], p[i]))
            fNew.write('\n')
    finally:
        fNew.close()
    runtime = run(configNum)
    configNum += 1
    return runtime


def draw_target(bo):
    plt.cla()
    plt.clf()
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
    plt.savefig(father_path + "/target.png")
    plt.show()


if __name__ == '__main__':
    # 读取重要参数
    vital_params = pd.read_csv(vital_params_path)
    print('重要参数列表（将贝叶斯的x_probe按照重要参数列表顺序转成配置文件实际运行:')
    print(vital_params)
    # 参数范围和精度，从参数范围表里面获取
    sparkConfRangeDf = pd.read_excel(conf_range_table)
    sparkConfRangeDf.set_index('SparkConf', inplace=True)
    confDict = sparkConfRangeDf.to_dict('index')

    '''
    2022/2/16
    设置初始样本点
    共采用8条数据作为初始样本点
    '''
    # dataset=gan_random(8)
    dataset=gan_random(args.initpoints)


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

    vital_params_name = sorted(d2)
    print('vital_params_name = ' + str(vital_params_name))
    vital_params_list = sorted(d2)
    vital_params_list.append('runtime')
    print('vital_params_list = ' + str(vital_params_list))
    # ------------------ 选择初始样本（3个方法选其一） start -------------
    # 选择所有样本
    initsamples = ganrs_samples_all(initsamples_df=dataset, vital_params_list=vital_params_list)
    # ------------------ 选择初始样本（3个方法选其一） end -------------

    bounds_transformer = SequentialDomainReductionTransformer()
    #定义贝叶斯优化模型
    optimizer = BayesianOptimization(
            f=black_box_function,
            pbounds=d2,
            verbose=2,
            random_state=1,
            bounds_transformer=bounds_transformer,
            custom_initsamples=initsamples
        )
    logpath = father_path + "/logs.json"
    logger = JSONLogger(path=logpath)
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    init_points = len(initsamples)
    n_iter = args.niters
    optimizer.maximize(init_points=init_points, n_iter=n_iter, acq='ei')
    print(optimizer.max)
    draw_target(optimizer)


    #存储数据
    # 读取json文件, 转成csv
    res_df = pd.DataFrame()
    for line in open(logpath).readlines():
        one_res = {}
        js_l = json.loads(line)
        one_res['target'] = -js_l['target']
        for pname in vital_params_name:
            one_res[pname] = js_l['params'][pname]
        df = pd.DataFrame(one_res, index=[0])
        res_df = res_df.append(df)
    # 设置索引从1开始
    res_df.index = range(1, len(res_df)+1)
    res_df.to_csv(generation_confs)
