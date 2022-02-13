import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="report")
parser.add_argument('-b', '--benchmark', help='name of benchmark', type=str)
parser.add_argument('-f', '--filename', help='input filename', type=str)

opts = parser.parse_args()

# dataframe
file_location = '/home/collect/data/os/' + opts.benchmark + '/' + opts.filename + '.csv'
csv_data = pd.read_csv(file_location, sep='\t', low_memory=False)  # 防止弹出警告 。
os_data = pd.DataFrame(csv_data)  # csv 转 dataframe

# 删除前4行数据
os_data = os_data.drop(os_data.index[:3])
os_data = os_data.T.reset_index(drop=True).T  # 重置列索引

os_data = pd.concat([os_data, os_data[0].str.split(',', expand=True)], axis=1)  # 列拆分，后合并
os_data = os_data.T.reset_index(drop=True).T  # 重置列索引
os_data = os_data.drop([0], axis=1)  # 删除第一列a

# 取出第一行和第二行
title_up = np.array(os_data[0:1]).flatten().tolist()  # 返回第1行到第2行的所有行
title_down = np.array(
    os_data[1:2]).flatten().tolist()  # 把pandas dataframe转为list方法：先用numpy的 array() 转为ndarray类型，再用tolist()函数转为list

# 去掉双引号
for i in range(0,len(title_up)):
    if ('"' in str(title_up[i])):
        title_up[i] = eval(str(title_up[i]))
    if ('"' in str(title_down[i])):
        title_down[i] = eval(str(title_down[i]))

# 两个列表对应位置拼接
new_title = []
for i in range(0, len(title_up)):
    if (str(title_up[i]) == 'nan' or len(str(title_up[i])) == 0):
        title_up[i] = str(title_up[i - 1])
    new_title.append(str(title_up[i]) + ' - ' + str(title_down[i]))

os_data = os_data.drop(os_data.index[:2])  # 删除前2行（原来的两行表头）
os_data.columns = new_title  # 修改列索引名为new_title
os_data.to_csv('/home/collect/csv/os/' + opts.benchmark + '/' + opts.filename + '.csv', index=False)
