import pandas as pd
import shutil
import random
import argparse

parser = argparse.ArgumentParser(description="generateRandomConfig")
parser.add_argument('-cn', '--configNum', help='the start num of config', type=int)
parser.add_argument('-cp', '--configPath', help='Configure storage path', type=str)
parser.add_argument('-p', '--parameters', help='path of parameters file which should use in the config', type=str)
parser.add_argument('-n', '--num', help='the num of config file should generate', type=int)
parser.add_argument('-t', '--type', help='the type of benchmark', type=str)
parser.add_argument('-r', '--range', help='the path of conf_range_table', type=str)
parser.add_argument('-a', '--all', help='whether contains all resource parameter', type=bool)
parser.add_argument('-m', '--memory', help='Upper limit of memory', type=int)

opts = parser.parse_args()


# 服务器运行spark时config文件
config_run_path = opts.configPath
# 重要参数
vital_params_path = opts.parameters
# 维护的参数-范围表
conf_range_table = opts.range

# 格式化参数配置：精度、单位等
def formatConf(conf, value):
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
            rangeList = confDict[conf]['Range'].split(' ')
            res = rangeList[int(res)].lower()
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

# 定义遗传算法的适应度函数
# 1、实际运行
configNum = opts.configNum
def schafferRun(p):

    global configNum
    # 打开配置文件模板
    fTemp = open('/usr/local/home/zwr/parameterSorting/configTemp_' + opts.type, 'r')
    # 复制模板，并追加配置
    fNew = open(config_run_path + '/config' + str(configNum), 'a+')
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
    configNum = configNum + 1
    return random.random()

def generate_random_num(min_value, max_value, pre):
    if pre == 1.0:
        return random.randint(int(min_value), int(max_value))
    elif pre == 0.01:
        return random.uniform(min_value, max_value)


# 读取重要参数
vital_params = pd.read_csv(vital_params_path)
# 参数范围和精度，从参数范围表里面获取
sparkConfRangeDf = pd.read_excel(conf_range_table)
sparkConfRangeDf.set_index('SparkConf', inplace=True)
confDict = sparkConfRangeDf.to_dict('index')

if opts.all:
    for i in range(opts.num):
        executorMemory = 0
        is_offHeap = 0
        offHeap = 0
        # 遍历训练数据中的参数，产生对应的随机数
        p = []  # 对应的取值
        for conf in vital_params['vital_params']:
            if conf == 'spark.executor.memory':
                executorMemory = generate_random_num(confDict[conf]['min'], (opts.memory - offHeap / 1024) / 1.12 * 1.0, confDict[conf]['pre'])
                p.append(executorMemory)
            elif conf == 'spark.memory.offHeap.enabled':
                is_offHeap = generate_random_num(confDict[conf]['min'], confDict[conf]['max'], confDict[conf]['pre'])
                p.append(is_offHeap)
            elif conf == 'spark.memory.offHeap.size':
                if is_offHeap == 1:
                    offHeap = generate_random_num(confDict[conf]['min'], confDict[conf]['max'], confDict[conf]['pre'])
                p.append(offHeap)
            elif conf == 'spark.executor.memoryOverhead':
                num = generate_random_num(executorMemory * 1024 * 0.06, executorMemory * 1024 * 0.12, confDict[conf]['pre'])
                p.append(max(num, 384))
            elif conf == 'spark.executor.cores':
                p.append(int(executorMemory / 2))
            elif conf in confDict:
                p.append(generate_random_num(confDict[conf]['min'], confDict[conf]['max'], confDict[conf]['pre']))
            else:
                print('-----该参数没有维护: ', conf, '-----')
        schafferRun(p)
else:
    for i in range(opts.num):
        # 遍历训练数据中的参数，产生对应的随机数
        p = []  # 对应的取值
        for conf in vital_params['vital_params']:
            if conf in confDict:
                p.append(generate_random_num(confDict[conf]['min'], confDict[conf]['max'], confDict[conf]['pre']))
            else:
                print('-----该参数没有维护: ', conf, '-----')
        schafferRun(p)