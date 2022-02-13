import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--benchmark', help='the benchmark name such as wordcount-20G-6000', type=str)
opts = parser.parse_args()

runtime_path = '/home/collect/runtime/' + opts.benchmark + '-runtime.txt'
config_path = '/usr/local/home/zwr/hibench-spark-config/' + opts.benchmark

runtimes = open(runtime_path).readlines()
configs = open(config_path + '/config1').readlines()
configs = configs[39:]

for i in range(len(configs)):
    configs[i] = configs[i].split()[0]
configs.append('runtime')

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


df = pd.DataFrame(index=range(1, len(runtimes) + 1), columns=configs, dtype='float64')

# 写入执行时间
for i in range(len(runtimes)):
    df.loc[i + 1].loc['runtime'] = float(runtimes[i].split()[4])


for i in range(1, len(runtimes) + 1):
    # 写入配置参数
    file_location = config_path + '/config' + str(i)
    lines = open(file_location, 'r').readlines()
    lines = lines[39:]
    for line in lines:
        result = line.split(" ")
        result[-1] = float(data_format(result[-1].strip()))
        df.loc[i].loc[result[1].split()[0]] = result[-1]
print(df.info())
df.to_csv('/usr/local/home/zwr/' + opts.benchmark + '-parameters_runtime.csv', index=False)
