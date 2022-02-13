import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="report")
parser.add_argument('-b', '--benchmark', help='name of benchmark', type=str)
parser.add_argument('-f', '--filename', help='input filename', type=str)
parser.add_argument('-e', '--eventslist', help='the path of eventlists',
                    default='/usr/local/home/yyq/events_hard_pmu.txt')
opts = parser.parse_args()

# 获得事件列表
def getEvnetsLists(eventsData):
    with open(eventsData) as events_file:
        allEvents = events_file.read().splitlines()
    eventsWithoutSpace = []  # 去掉事件左右两边的空格
    for e in allEvents:
        eventsWithoutSpace.append(e.strip())
    eventsWithoutSpace = sorted(set(eventsWithoutSpace), key=eventsWithoutSpace.index)  # 删除列表中的重复元素并且生成一个新的排好序的list对象
    return eventsWithoutSpace


# 获得事件总数
def getEvnetsNum(eventsData):
    with open(eventsData) as events_file:
        allEvents = events_file.read().splitlines()
    eventsWithoutSpace = []  # 去掉事件左右两边的空格
    for e in allEvents:
        eventsWithoutSpace.append(e.strip())
    eventsWithoutSpace = sorted(set(eventsWithoutSpace), key=eventsWithoutSpace.index)  # 删除列表中的重复元素且不改变顺序
    return len(eventsWithoutSpace)


eventsData = opts.eventslist
eventlists = getEvnetsLists(eventsData)
eventsnum = getEvnetsNum(eventsData)

# dataframe
time = []
events = []
values = []
# units = []
file_location = '/home/collect/data/micro/' + opts.benchmark + '/' + opts.filename + '.dat'
lines = open(file_location, 'r').readlines()
del lines[0:3]
for i in range(len(lines)):  # [0] is perf conmmand [1] is note [2] is '\n'
    time.append(lines[i].split(',')[0])
    events.append(lines[i].split(',')[3])
    values.append(lines[i].split(',')[6])
    # units.append(lines[i].split(',')[7])
test_1 = pd.DataFrame(events, time)  # index=list,column=list

newUnits = []

test_1[1] = values


def ab(df):
    return ','.join(df.values)


test_1 = test_1.groupby([0])[1].apply(ab)
test_1 = test_1.reset_index()

test_1 = pd.concat([test_1, test_1[1].str.split(',', expand=True)], axis=1)  # [值放在一起]列拆分成多个值列，后合并为同一个dataframe
test_1 = test_1.T.reset_index().T  # 重置列索引
test_1.drop(['index'], axis=0, inplace=True)  # 删除转置后重置索引多出来的index行

test_1.drop([1], axis=1, inplace=True)  # 删除第二列 [值放在一起]列
test_1 = test_1.T.reset_index().T  # 重置列索引
test_1.drop(['index'], axis=0, inplace=True)  # 删除转置后重置索引多出来的index行

test_1.index = test_1[0]  # 将事件名列作为index行索引
test_1 = test_1.loc[eventlists]  # 将index索引按照事件列表的顺序进行排序

res = test_1.T  # 转置行变列，事件索引变成列索引

res.drop([1], axis=0, inplace=True)  # drop first 2 second
res.drop([2], axis=0, inplace=True)  # node first ’perf‘ & after sleep 1，master can work
res = res.reset_index(drop=True)  # 重置索引index，并且删除原索引

# 将instructions移到最后一列 ， 并改名为IPC
temp_IPC_data = res['instructions']  # 取出instructions列
IPC_data = temp_IPC_data.to_frame()  # 将格式转为dataframe
IPC = IPC_data.reset_index(drop=True)  # 重置索引index，并且删除原索引
IPC.rename(columns={'instructions': 'IPC'}, inplace=True)  # 改名列索引为IPC

res = pd.concat([res, IPC], axis=1).drop(['instructions'], axis=1)  # 列尾加上IPC，并删除res = pd.concat([res, IPC], axis=1).drop(['instructions'],axis=1) # 列尾加上IPC，并删除

res = res.T.reset_index(drop=True).T  # 重新设置列索引 从0开始

# 把dataframe文件转为csv文件
res.to_csv('/home/collect/csv/micro/' + opts.benchmark + '/' + opts.filename + '.csv', index=False, header=None)

