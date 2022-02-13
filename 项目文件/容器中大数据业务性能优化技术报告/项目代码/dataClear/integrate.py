import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="A System Metric Monitor.")
parser.add_argument('-b', '--benchmark', help='name of benchmark', type=str)
parser.add_argument('-n', '--number', help='The number of files that need to be integrated', type=int)
opts = parser.parse_args()


# 均匀的删除df中指定数量的行
def delete(num_row, df):
    if num_row == 0:
        return
    df_row = df.shape[0]
    new_df_row = df_row - num_row
    distance = df_row * 1.0 / num_row
    for i in np.arange(distance - 1, df_row + 1, distance):
        delete_row = round(i)
        df.drop(delete_row, inplace=True)
    df.set_index(pd.Index(range(0, new_df_row, 1)), inplace=True)


for i in range(1, opts.number + 1):
    print('============' + opts.benchmark + '/int-' + str(i) + '-report.csv' + '============')
    # 读取三层所采集到的数据
    df_os = pd.read_csv("/home/collect/csv/after_clear/os/" + opts.benchmark + "/os-" + str(i) + "-report.csv")
    df_container = pd.read_csv("/home/collect/csv/after_clear/container/" + opts.benchmark + "/co-" + str(i) + "-report.csv")
    df_micro = pd.read_csv("/home/collect/csv/after_clear/micro/" + opts.benchmark + "/micro-" + str(i) + "-report.csv")

    # 获取每个df的行号
    rows_micro = df_micro.shape[0]
    rows_os = df_os.shape[0]
    rows_container = df_container.shape[0]

    # 找出最小的行号
    min_row = min(rows_container, rows_os, rows_micro)

    # 删除一定数量的行号，使得所有的df的行号等于最小的行号
    delete(rows_micro - min_row, df_micro)
    delete(rows_os - min_row, df_os)
    delete(rows_container - min_row, df_container)

    # 将所有的df拼接在一起，并写到指定位置
    res = pd.concat([df_os, df_container, df_micro], axis=1)
    res.to_csv("/home/collect/csv/after_clear/integration/" + opts.benchmark + "/int-" + str(i) + "-report.csv", index=False)
