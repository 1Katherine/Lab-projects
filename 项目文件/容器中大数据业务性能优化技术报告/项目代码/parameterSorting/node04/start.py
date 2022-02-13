import os
import argparse
import pandas as pd
import time

parser = argparse.ArgumentParser(description="parameterSorting")
parser.add_argument('-b', '--benchmark', help='the benchmark name such as wordcount-20G-6000', type=str)
parser.add_argument('-m', '--model', help='the model using in parameterSorting', type=str)
parser.add_argument('-ucn', '--upConfigNum', help='the start num of config', type=int)
parser.add_argument('-dcn', '--downConfigNum', help='the start num of config', type=int)
parser.add_argument('-t', '--type', help='the type of benchmark', type=str)
parser.add_argument('-udf', '--upDataFrame', help='Whether to create a new dataFrame for stage one', type=bool, default=False)
parser.add_argument('-ddf', '--downDataFrame', help='Whether to create a new dataFrame for stage two', type=bool, default=False)
parser.add_argument('-mm', '--maxMemory', help='Upper limit of memory', type=int)

opts = parser.parse_args()

modelList = ['lgb', 'gdbt', 'ada', 'xgb', 'rf']

# 转换配置文件的格式
def data_format(data):
    if data == 'false':
        return 0
    elif data == 'true':
        return 1
    elif data == 'FIFO':
        return 0
    elif data == 'FAIR':
        return 1
    elif data == 'org.apache.spark.serializer.JavaSerializer':
        return 0
    elif data == 'org.apache.spark.serializer.KryoSerializer':
        return 1
    elif data == 'snappy':
        return 0
    elif data == 'lzf':
        return 1
    elif data == 'lz4':
        return 2
    elif 'min' in data:
        return data[:-3]
    elif 'ms' in data:
        return data[:-2]
    elif 's' in data or 'm' in data or 'g' in data or 'k' in data:
        return data[:-1]
    else:
        return data

lastRunTime = 0.01

def hibench_runtime():
    get_time_cmd = 'tail -n 1 /usr/local/home/hibench/hibench/report/hibench.report'
    line = os.popen(get_time_cmd).read()
    return float(line.split()[4])

def tpcds_runtime(path):
    global lastRunTime
    is_find = os.path.exists(path)
    if is_find == False:
        return lastRunTime
    runtimes = open(path).readlines()
    for i in range(0, len(runtimes)):
        if 'Time' in runtimes[i]:
            startTime = float(runtimes[i].split(':')[1])
            endTime = float(runtimes[i + 1].split(':')[1])
            return endTime - startTime
        runtime = float(runtimes[i].split()[0])
        if runtime == 2000:
            return lastRunTime
    return lastRunTime

# 配置文件存放路径，运行基准，误差要求，配置参数列表文件存放位置
def run(configPath, benchmark, min_error, parameter, logfile, configNum, py, other, type, is_creat, all):
    global opts
    global modelList
    global lastRunTime
    # 建立初始的空CSV文件
    configs = open(parameter).readlines()
    configs = configs[1:]

    for i in range(len(configs)):
        configs[i] = configs[i].split()[0]
    configs.append('runtime')

    if is_creat == True:
        df = pd.DataFrame(columns=configs, dtype='float64')
        df.to_csv('/home/collect/csv/trainData/' + benchmark + '-parameters_runtime.csv', index=False)

    # 找出随机配置出现的行号
    lines = open('/usr/local/home/zwr/parameterSorting/configTemp_' + type).readlines()
    rp = len(lines)

    # 建立配置文件存放目录
    os.system('mkdir -p ' + configPath)
    os.system('mkdir -p /usr/local/home/zwr/hibench-spark-config/errorConfig/' + benchmark)
    # 建立运行时间的存放目录
    if type == 'tpcds':
        os.system('mkdir -p /home/collect/runtime/tpcds/' + benchmark)
    global_error = 1.0
    global_model = 'default'
    global_configNum = 0
    global_round = 0
    while configNum < 2000:
        df = pd.DataFrame(index=range(1, 6), columns=configs, dtype='float64')
        # 生成配置文件
        os.system('python3 /usr/local/home/zwr/parameterSorting/configGenerate.py -cn=' + str(configNum) +
                  ' -cp=' + configPath +
                  ' -p=' + parameter +
                  ' -n=5' +
                  ' -t=' + type +
                  ' -r=' + "/usr/local/home/zwr/parameterSorting/Spark_conf_range_" + opts.benchmark.split('-')[0] + ".xlsx" +
                  ' -a=' + str(all) +
                  ' -m=' + str(opts.maxMemory))
        i = 1
        while i <= 5:
            # 使用指定配置文件运行指定基准
            if type == 'hibench':
                os.system('/usr/local/home/zwr/parameterSorting/' + opts.benchmark + '-sort.sh ' + str(configNum) + ' ' + configPath
                          + ' >> ' + '/usr/local/home/zwr/log/' + opts.benchmark + '-runningOutPut.log')
            elif type == 'tpcds':
                os.system('/usr/local/home/zwr/parameterSorting/' + opts.benchmark + '-sort.sh ' + str(configNum) + ' ' + configPath
                          + " " + benchmark + ' >> ' + '/usr/local/home/zwr/log/' + opts.benchmark + '-runningOutPut.log')

            # 睡眠5秒，保证hibench.report文件完成更新后再读取运行时间
            time.sleep(5)
            # 获取此次spark程序的运行时间
            if type == 'hibench':
                runtime = hibench_runtime()
            elif type == 'tpcds':
                path = '/home/collect/runtime/tpcds/' + benchmark + '/' + 'config' + str(configNum) + '.result'
                runtime = tpcds_runtime(path)
                if runtime == lastRunTime:
                    os.system('rm -rf ' + path)

            # 检测到上次的运行时间与此次相同则说明程序运行失败，就重新生成配置文件运行
            if lastRunTime == runtime:
                os.system('cat ' + configPath + '/config' + str(configNum) +
                          ' >> /usr/local/home/zwr/hibench-spark-config/errorConfig/' + benchmark + '/config' + str(configNum))
                os.system('echo ========================================= >> /usr/local/home/zwr/hibench-spark-config/errorConfig/' + benchmark + '/config' + str(configNum))
                os.system('rm -rf ' + configPath + '/config' + str(configNum))
                os.system('python3 /usr/local/home/zwr/parameterSorting/configGenerate.py -cn=' + str(configNum) +
                          ' -cp=' + configPath +
                          ' -p=' + parameter +
                          ' -n=1' +
                          ' -t=' + type +
                          ' -r=' + "/usr/local/home/zwr/parameterSorting/Spark_conf_range_" + opts.benchmark.split('-')[0] + ".xlsx" +
                          ' -a=' + str(all) +
                          ' -m=' + str(opts.maxMemory))
                continue
            lastRunTime = runtime
            file_location = configPath + '/config' + str(configNum)
            lines = open(file_location, 'r').readlines()
            lines = lines[rp:]
            for line in lines:
                result = line.split()
                result[-1] = float(data_format(result[-1].strip()))
                df.loc[i].loc[result[0]] = result[-1]
            df.loc[i].loc['runtime'] = runtime
            i = i + 1
            configNum = configNum + 1

        # 将新的数据追加到csv中
        df.to_csv('/home/collect/csv/trainData/' + benchmark + '-parameters_runtime.csv', index=False, mode='a',
                  header=0)
        if configNum > 100:
            file = open(logfile, 'a+')
            file.write('*******************************************************\n')
            file.write("执行了" + str(configNum - 1) + "个配置文件\n")
            file.write('*******************************************************\n')
            file.close()
            local_error = 1.0
            local_model = 'default'

            for model in modelList:
                # 建立目录存放训练结果
                os.system('mkdir -p /home/collect/trainResult/' + benchmark + '/files' + str(configNum - 1) + '/' + model)
                # 进行模型的训练
                os.system(
                    'python3 /usr/local/home/zwr/parameterSorting/' + py + ' -n ' + model + other +
                    ' -s /home/collect/trainResult/' + benchmark + '/files' + str(configNum - 1) + '/' + model + '/' +
                    ' -f /home/collect/csv/trainData/' + benchmark + '-parameters_runtime.csv')

                # 读取误差
                get_error_cmd = 'tail -n 1 /home/collect/trainResult/' + benchmark + '/files' + str(configNum - 1) + '/' + model + '/parameters_error.txt'
                line = os.popen(get_error_cmd).read()
                error = float(line.split()[0])
                # 记录日志
                file = open(logfile, 'a+')
                date = os.popen('date').read()
                file.write('====================================\n')
                file.write(date)
                file.write("使用" + model + "进行建模，误差为" + str(error))
                file.write('\n')
                file.write('====================================\n')
                file.close()
                if error < local_error:
                    local_error = error
                    local_model = model

            if local_error < global_error:
                global_error = local_error
                global_configNum = configNum
                global_model = local_model
                global_round = 0
            else:
                global_round = global_round + 1
            # 误差满足条件就终止
            if global_error < min_error or global_round == 5:
                return global_configNum, global_error, global_model


# 第一次运行找出重要参数
result = run('/usr/local/home/zwr/hibench-spark-config/' + opts.benchmark + '-sorting',
             opts.benchmark + '-sorting', 0.15,
             '/usr/local/home/zwr/parameterSorting/parameters_set_' + opts.type + '.txt',
             '/usr/local/home/zwr/log/' + opts.benchmark + '-parameterSorting.log', opts.upConfigNum,
             'main_3.py',
             ' -step 1 -left 5 -t runtime -p parameters',
             opts.type,
             opts.upDataFrame,
             True)
(configNum, error, model) = result
fileObject = open('/usr/local/home/zwr/log/' + opts.benchmark + '-parameterSorting.log', 'a+')
date = os.popen('date').read()
fileObject.write('#########################################################\n')
fileObject.write(date)
fileObject.write("执行了" + str(configNum - 1) + "个配置文件，使用" + model + "筛选出了重要参数，误差为" + str(error) + "\n")
fileObject.write('#########################################################\n')
fileObject.close()

# 第二次运行建立重要参数到执行时间的预测模型
# result = run('/usr/local/home/zwr/hibench-spark-config/' + opts.benchmark + '-important_parameter_runtime',
#              opts.benchmark + '-important_parameter_runtime', 0.1,
#              '/home/collect/trainResult/' + opts.benchmark + '-sorting' + '/files' + str(configNum - 1) + '/' + model + '/selected_parameters.txt',
#              '/usr/local/home/zwr/log/' + opts.benchmark + '-parameterSorting.log', opts.downConfigNum,
#              'main_time_predict.py',
#              ' -t runtime',
#              opts.type,
#              opts.downDataFrame,
#              False)
# (configNum_best, error_best, model_best) = result
# fileObject = open('/usr/local/home/zwr/log/' + opts.benchmark + '-parameterSorting.log', 'a+')
# date = os.popen('date').read()
# fileObject.write('#########################################################\n')
# fileObject.write(date)
# fileObject.write("执行了" + str(configNum_best - 1) + "个配置文件，使用" + model_best + "产生了重要参数到执行时间的预测模型，误差为" + str(error_best) + "\n")
# fileObject.write('#########################################################\n')
# fileObject.close()