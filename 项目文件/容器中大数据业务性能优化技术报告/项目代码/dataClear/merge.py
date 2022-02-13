import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser(description="A System Metric Monitor.")
parser.add_argument('-b', '--benchmark', help='name of benchmark', type=str)
parser.add_argument('-t', '--type', help='The type of data need to be merger', type=str)
opts = parser.parse_args()


# 合并为一个csv
def merge_results(csvPath, resultPath):
    # path = './wordcount-205G/clearedData'  # 设置csv所在文件夹
    files = os.listdir(csvPath)  # 获取文件夹下所有文件名

    df1 = pd.read_csv(csvPath + '/' + files[0], encoding='gbk')  # 读取首个csv文件，保存到df1中

    for file in files[1:]:
        df2 = pd.read_csv(csvPath + '/' + file, encoding='gbk')  # 打开csv文件，注意编码问题，保存到df2中
        df1 = pd.concat([df1, df2], axis=0, ignore_index=True)  # 将df2数据与df1合并

    df1 = df1.drop_duplicates()  # 去重
    df1 = df1.reset_index(drop=True)  # 重新生成index
    # df1 = df1.loc[:, :'instructions']  # 将instructions后面的数据都清理掉
    df1.fillna(0,inplace=True)
    df1.to_csv(resultPath, index=False)  # 将结果保存为新的csv文件
    # df1.to_csv(csvPath + '/' + 'total.csv', index=False)  # 将结果保存为新的csv文件

# 为合并完成的数据建立存放目录
cmd = '/usr/local/home/zwr/checkDirectory.sh ' + '/home/collect/train_data/' + opts.type + '/' + opts.benchmark.split('-')[0]
os.system(cmd)
merge_results('/home/collect/csv/after_clear/' + opts.type + '/' + opts.benchmark + '/',
              '/home/collect/train_data/' + opts.type + '/' + opts.benchmark.split('-')[0] + '/' + opts.type + '-' +
              opts.benchmark + '.csv')
