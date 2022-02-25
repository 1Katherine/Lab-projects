'''
2022/2/16
均为不同的取样本的方式
'''
# 取所有样本作为bo初始样本
def ganrs_samples_all(initsamples_df, vital_params_list):
    # 初始样本
    initsamples = initsamples_df[vital_params_list].to_numpy()
    print('选择50%rs和50%gan的所有样本作为bo算法的初始样本,样本个数为:' + str(len(initsamples)))
    return initsamples

# 获取dataframe的前n行样本作为初始样本
def get_head_n(n,initsamples_df, vital_params_list):
    print('取出前' + str(n) + '个样本')
    initsamples_head = initsamples_df.head(n)
    initsamples = initsamples_head[vital_params_list].to_numpy()
    print('取出前两组样本作为初始样本：, shape = ' + str(initsamples.shape))
    return initsamples

# 每隔n行取一行
def get_ganrs_intevaln(n,initsamples_df, vital_params_list):
    a = []
    for i in range(0, len(initsamples_df), n):  ##每隔86行取数据
        a.append(i)
    sample = initsamples_df.iloc[a]
    initsamples = sample[vital_params_list].to_numpy()
    print('间隔采样，取出的行号为：' + str(a) + ' , shape = ' + str(initsamples.shape))
    return initsamples

# 样本按照runtime 升序排序，获取runtime最少的前n个样本作为初始样本
def get_best_n(n,initsamples_df, vital_params_list):
    initsamples_sort = initsamples_df.sort_values(by='runtime', ascending=True)
    initsamples_head = initsamples_sort[vital_params_list].head(n)
    initsamples = initsamples_head.to_numpy()
    print('把执行时间最少的前几个样本作为初始样本，shape=' + str(initsamples.shape))
    return initsamples

# if __name__ == '__main__':
    # ------------------ 选择初始样本（3个方法选其一） start -------------
    # if sample_type == 'all':
    #     # 选择所有样本
    #     initsamples = ganrs_samples_all(initsamples_df=dataset)
    # elif sample_type == 'firstngroup':
    #     # 选择前n个样本,根据实际情况设置样本
    #     initsamples = get_head_n(initsamples_df=dataset,n=0)
    # elif sample_type == 'interval':
    #     # 每隔3个样本选择一个样本（包括第三个样本）
    #     initsamples = get_ganrs_intevaln(initsamples_df=dataset,n = 0)
    # elif sample_type == 'best':
    #     initsamples = get_best_n(initsamples_df=dataset,n=8)
    # else:
    #     raise Exception("[!] 请在all、firstngroup、interval、best中选择一种初始样本方式，firstngroup表示前n组样本，"
    #                     "interval表示间隔采样，best表示执行时间最少前几个样本")