import pandas as pd
import time
import warnings
import shutil
import matplotlib.pyplot as plt
import numpy as np
from hyperopt import tpe, hp, fmin, Trials
import os
import argparse
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--benchmark', type=str, help='benchmark type')
parser.add_argument('--max_evals', type=int, help='The number of iterations of the Bayesian optimization algorithm')
args = parser.parse_args()
print('benchmark = ' + args.benchmark + '\t bo迭代搜索次数：--max_evals = ' + str(args.max_evals))

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
generation_confs = father_path +  "/generationConf.csv"

warnings.filterwarnings("ignore")

def draw_target(loss):
    np_loss = np.array(loss)
    # 画图
    plt.plot(loss, label='hyperopt max_evals = ' + str(args.max_evals))
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
    print('configNum = ' + str(configNum) + '\t run_cmd = ' + str(run_cmd) + '\t runtime = ' + str(runtime))
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


def objective(params):
    print(params)
    i = []
    for conf in vital_params['vital_params']:
        i.append(params[conf])
    print('objective中schafferRun(i) 中的config为' + str(i))
    return schafferRun(i)


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
        获取space格式 传入space = {
                            'x': hp.uniform('x', -6, 6),
                            'y': hp.uniform('y', -6, 6)
                        }
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
            space.update(d1)
        else:
            print(conf, '-----参数没有维护: ', '-----')

    print('space:' + str(space))
    trials = Trials()
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=args.max_evals,
        trials=trials
    )

    print('best\n' + str(best))

    result = []
    loss = []
    for i in range(args.max_evals):
        res = {}
        for key in space:
            res[key] = trials.vals[key][i]
        res['loss'] = trials.results[i]
        loss.append(trials.results[i]['loss'])
        print(res)
        print(res)
        result.append(res)
    print('result\n' + str(result))

    logpath = "output.txt"
    f = open(logpath, 'w+')
    for i, r in enumerate(result):
        print(str(i) + "  " + str(r), file=f)
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
        for pname in vital_params['vital_params']:
            one_res[pname] = js_l[pname]
        df = pd.DataFrame(one_res, index=[0])
        res_df = res_df.append(df)
    # 设置索引从1开始
    res_df.index = range(1, len(res_df) + 1)
    res_df.to_csv(generation_confs)