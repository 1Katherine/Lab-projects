import pandas as pd

# --------------------- 生成 gan-rs 初始种群 start -------------------
# 选择所有初始样本
def ganrs_samples_all():
    # 初始样本
    initsamples = initsamples_df.to_numpy()
    return initsamples

# 获取dataframe的前n行样本作为初始样本
def get_head_n(n):
    initsamples_head = initsamples_df.head(n)
    initsamples = initsamples_head.to_numpy()
    return initsamples

# 每隔n行取一行
def get_ganrs_intevaln(n):
    a = []
    for i in range(0, len(initsamples_df), n):  ##每隔86行取数据
        a.append(i)
    print('取出的行号为：' + str(a))
    initsamples = initsamples_df.iloc[a].to_numpy()
    return initsamples

# 样本按照runtime 升序排序，获取runtime最少的前n个样本作为初始样本
def get_best_n(n):
    initsamples_sort = initsamples_df.sort_values(by='runtime', ascending=True)
    initsamples_head = initsamples_sort.head(n)
    initsamples = initsamples_head.to_numpy()
    return initsamples
# --------------------- 生成 gan-rs 初始种群 end -------------------

if __name__ == '__main__':
    initpoint_path = 'wordcount-100G-GAN-30.csv'
    initsamples_df = pd.read_csv(initpoint_path)
    ganrs_group = 6

    # # 所有样本
    # ganrs_samples = ganrs_samples_all()
    # print(ganrs_samples)
    # print(ganrs_samples.shape)
    #
    # # 获取dataframe的前n行样本作为初始样本
    # # # gan和rs的一组采样间隔是6，每采样3个随机样本就会采样3个GAN样本
    # headn = ganrs_group * 2
    # ganrs_samples_head = get_head_n(n=headn)
    # print(ganrs_samples_head)
    # print(ganrs_samples_head.shape)
    #
    #
    # # 从第0行开始每隔n行取一行数据(包括第n行）【0，3，2*3，3*3...】
    # ganrs_interval = ganrs_group / 2
    # ganrs_samples_interval = get_ganrs_intevaln(n=3)
    # print(ganrs_samples_interval)
    # print(ganrs_samples_interval.shape)

    ganrs_sort = get_best_n(n=10)
    print(ganrs_sort)
    print(ganrs_sort.shape)