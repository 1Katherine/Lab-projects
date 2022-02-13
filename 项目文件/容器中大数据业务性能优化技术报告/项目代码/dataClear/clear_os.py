from sklearn.impute import KNNImputer
import scipy.stats as st
import pandas as pd
import numpy as np
import argparse
import warnings


warnings.filterwarnings("ignore")  # 消除警告

parser = argparse.ArgumentParser(description="A System Metric Monitor.")
parser.add_argument('-b', '--benchmark', help='name of benchmark', type=str)
parser.add_argument('-n', '--number', help='The number of files that need to be cleaned', type=int)
opts = parser.parse_args()


# 合并两个节点的数据
def merge_nodes(node2FileName, node3FileName):
    df1 = pd.read_csv(node2FileName, encoding="utf-8")
    df2 = pd.read_csv(node3FileName, encoding="utf-8")
    df = df1.add(df2)
    df[df.select_dtypes(include=['number']).columns] *= 0.5
    df.dropna(axis=0, how='all', inplace=True)
    return df


# 清洗数据
def clear_data(df, resultPath):
    # 删除 整列都是 NAN 的指标列
    df.dropna(axis=1, how='all', inplace=True)
    # 删除 整列都是 0 的指标列
    df = df.loc[:, (df != 0).any(axis=0)]
    # 遍历所有的events
    for index, line in df.iteritems():
        # print(line)
        # event series
        line = pd.to_numeric(line)
        lineMax = line.max()
        lineMin = line.min()
        countZeros = 0
        # 1、设置 event series的 n, n值根据 event series的分布状况来设置 Gumbel和 logistic
        staticRes = st.normaltest(line)
        if staticRes.pvalue < 0.01:
            n = 5
        else:
            n = 3
        # 2、计算event series的 threshold
        threshold1 = line.mean() + n * line.std()
        threshold2 = line.mean() - n * line.std()
        # 3、间隔区间
        # L = (lineMax - lineMin) / round(math.sqrt(line.count()), 0)

        # 4、遍历event series替换离群值
        for i, item in line.iteritems():
            if item > threshold1 or item < threshold2:
                # print('下标：', i, '离群值：', line[index])
                # 设置为nan然后去填充
                line[i] = np.nan
                # print('下标：', i, '修改后：', line[index])
            if item == 0:
                countZeros += 1
        # event series的最大值小于0.01并且最小值为 0，则 0 值可以看作是正常值；
        if lineMax > 0.01:
            # 百分之八十都是 0 的列直接 drop
            if countZeros / line.count() > 0.8:
                df.drop(columns=index, inplace=True)
            # 其他包含 0 的数设置为 np.nan 用于填充缺失值
            elif countZeros > 0:
                df.loc[df[index] == 0, index] = np.nan

    # 使用 knn填充缺失值： k为3，缺失值为 0
    imp = KNNImputer(missing_values=np.nan, n_neighbors=3)
    dfColumn = df.columns
    outDf = pd.DataFrame(imp.fit_transform(df), columns=dfColumn)
    # 删除 整列都是 0 的指标列
    outDf = outDf.loc[:, (outDf != 0).any(axis=0)]
    outDf.to_csv(resultPath, index=False)


def clear(totalNum):
    for i in range(totalNum):
        print('============' + opts.benchmark + '/os-' + str(i + 1) + '-report.csv' + '============')
        fileNameStr1 = '/home/collect/csv/before_clear/os/node02/' + opts.benchmark + '/osmon-' + str(i + 1) + '-k8s-node02.csv'
        fileNameStr2 = '/home/collect/csv/before_clear/os/node03/' + opts.benchmark + '/osmon-' + str(i + 1) + '-k8s-node03.csv'
        df = merge_nodes(fileNameStr1, fileNameStr2)
        clear_data(df, '/home/collect/csv/after_clear/os/' + opts.benchmark + '/os-' + str(i + 1) + '-report.csv')

clear(opts.number)
