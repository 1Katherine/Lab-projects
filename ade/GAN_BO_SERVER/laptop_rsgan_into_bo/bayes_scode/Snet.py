import shutil
import torch
import numpy as np
from .Dataset import load_dataloader, dataset_to_below_1
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from .model import generator_module, discriminator_module
from .configuration import parser
import time
import torch.nn.functional as F
import random
from sklearn.utils import shuffle
import  os
'''
2022/1/23
计算余弦相似度
'''
#打印网络结构
def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

def write_DRData(data, save_path):  # 写入重要配置参数
    file = open(save_path + "Discriminator_rdata.txt", mode="a+")
    file.write(data)
    file.write("\n")
    file.close()
def write_time(data, save_path):  # 写入重要配置参数
    file = open(save_path + "run_data.txt", mode="a+")
    file.write(data)
    file.write("\n")
    file.close()

def write_epoch(data, save_path):  # 写入重要配置参数
    file = open(save_path + "run_epoch.txt", mode="a+")
    file.write(data)
    file.write("\n")
    file.close()
def cos_sim(a, b):
    a = np.array(a)
    a = a.reshape(-1)
    b = np.array(b)
    b = b.reshape(-1)
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    # print(a_norm)
    # print(b_norm)
    cos = (np.dot(a, b) / (a_norm * b_norm))
    return cos


'''
2022/2/16
添加欧式距离
'''


def ed(a, b):
    a = np.array(a)
    a = a.reshape(-1)
    b = np.array(b)
    b = b.reshape(-1)
    distance = np.sqrt(np.sum(np.square(a - b)))
    return distance


def formatConf(confDict, conf, value):
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
            res = str(bool(res)).lower()
        # 列表形式的参数（spark.serializer、spark.io.compression.codec等）
        elif confDict[conf]['unit'] == 'list':
            rangeList = confDict[conf]['Range'].split(' ')
            res = rangeList[res]
        # 拼接上单位
        else:
            res = str(res) + confDict[conf]['unit']
    else:
        res = str(res)
    return res


'''
2022/2/16
生成cosgan采样数据

'''


def train(df, first_time, args):
    # 标记开始时间
    start_time = time.time()

    # 导入数据
    sparkConfRangeDf = pd.read_excel(args.config_range)
    sparkConfRangeDf.set_index('SparkConf', inplace=True)
    datasets = dataset_to_below_1(args, df, sparkConfRangeDf)
    # 转换后的数据为
    try:
        if (df['spark.memory.offHeap.size'].item() == 0):
            datasets['spark.memory.offHeap.size'] = 0
    except:
        print('')

    # print(datasets)

    # df=df.iloc[:,-20:]
    samples, number_Features = datasets.shape
    args.number_features = number_Features
    args.cell_type = 'DNN'
    args.d_size = 1

    # print('traindata.shape:{}'.format(datasets.shape))
    count_value = sum(sum(abs(datasets.values)))
    mean_value = count_value / datasets.shape[1]
    # print('traindata.count_value:{}'.format(count_value))
    train_loader = load_dataloader(datasets, args.batch_size, args)
    generator_mod = generator_module(args)
    discriminator_mod = discriminator_module(args)
    optimizer_discriminator = torch.optim.Adam(discriminator_mod.parameters(), lr=args.learning_rate)
    optimizer_generator = torch.optim.Adam(generator_mod.parameters(), lr=args.learning_rate)
    loss_function = nn.BCELoss()

    # 画图，存储loss
    dec_loss = list()
    gen_loss = list()
    # print('---------- Networks architecture -------------')
    # print('')
    # print('---------- Generator architecture -------------')
    # print_network(generator_mod)
    # print('---------- Discriminator architecture -------------')
    # print_network(discriminator_mod)
    # print('-----------------------------------------------')
    m = 0
    '''
    2022/2/19
    k为期望的数据变化范围
    '''
    #k为期望的样本变化幅度
    k=0.25
    a = -1
    b = 1
    n = 0
    # 选取多少维数据
    means = []
    index = []

    for indexs, row in df.iteritems():
        index = np.append(index, indexs)
    index1 = np.append(index, 'range')
    index1 = np.append(index1, 'dissimilarity_value')
    '''
        2022/2/12
        加入cos_distance距离

        '''
    index1 = np.append(index1, 'cos_distance')
    '''
        2022/2/16
        加入欧式距离,与总的距离度量

        '''
    index1 = np.append(index1, ['Euclidean_distance', 'distance','discriminator_label'])
    final_data = pd.DataFrame(columns=index1)
    generated_data = pd.DataFrame(columns=index)
    for epoch in range(args.epochs):
        # print(epoch)
        # print(args.epochs)
        for n, (real_samples, _) in enumerate(train_loader):
            # print(1)
            real_samples_labels = torch.ones((args.batch_size, 1), device=torch.device(args.device))
            latent_space_samples = torch.randn((args.batch_size,  args.number_features),
                                               device=torch.device(args.device))

            generated_samples = generator_mod(latent_space_samples)
            generated_samples_labels = torch.zeros((args.batch_size, 1), device=torch.device(args.device))
            all_samples = torch.cat((real_samples.to(device=torch.device(args.device)),
                                     generated_samples.to(device=torch.device(args.device))))
            all_samples_labels = torch.cat((real_samples_labels, generated_samples_labels)).to(
                device=torch.device(args.device))

            # Training the discriminator
            discriminator_mod.zero_grad()
            output_discriminator = discriminator_mod(all_samples.float())
            loss_discriminator = loss_function(output_discriminator, all_samples_labels)
            loss_discriminator.backward()
            optimizer_discriminator.step()

            # Data for training the generator
            latent_space_samples = torch.randn((args.batch_size,  args.number_features),
                                               device=torch.device(args.device))
            # Training the generator
            generator_mod.zero_grad()
            generated_samples = generator_mod(latent_space_samples.float())
            '''
            利用距离指标来衡量
            '''
            # loss_generator = sum(sum(abs(generated_samples - real_samples)))
            ''' 
            2022/2/18
            变更loss
            '''
            loss_generator = sum(sum(abs((generated_samples - real_samples)/2*mean_value)))
            loss_generator.backward()
            optimizer_generator.step()

            # Show loss
            if epoch % 1 == 0:
                # print(f"Epoch: {epoch} Loss G.: {loss_generator}")
                with torch.no_grad():
                    dec_loss.append(loss_discriminator)
                    gen_loss.append(loss_generator)
                torch.save(generator_mod, "GAN_generator.pth")
        '''
         2022/1/21
         添加指标衡量样本之间的远近
         一个样本随机性太强，增加为10个样本
         '''
        generate_samples = torch.randn((10, args.d_size * args.number_features),
                                       device=torch.device(args.device)).clone().detach()
        # distance = sum(sum(abs(generator_mod(generate_samples) - real_samples))) / 10
        '''
        2022/2/18
       早停指标变更
        '''
        distance = sum(sum(abs((generator_mod(generate_samples) - real_samples)/2*mean_value))) / 10
        # print(distance)

        # distance=number_Features
        # if(epoch%50==0):
        #     distance=0
        #     generate_samples = torch.randn((10, args.d_size * args.number_features),
        #                                        device=torch.device(args.device))
        #
        #     print(real_samples.T[1])
        #     print(generator_mod(generate_samples).T[1][1])
        #     for i in range(real_samples.shape[1]):
        #         for j in range(10):
        #             if real_samples.T[i]==0:
        #                 distance=distance+abs(generator_mod(generate_samples).T[i][j] - real_samples.T[i])
        #             else:
        #                 distance=distance+abs((generator_mod(generate_samples).T[i][j] - real_samples.T[i])/real_samples.T[i])
        #     distance=distance/10
        #     print(distance)


        # if distance<number_Features * k:
        #     latent_space_samples = torch.randn((10, args.number_features), device=torch.device(args.device)).clone().detach()
        #     generated_samples2 = generator_mod(latent_space_samples)
        #     generated_samples2 = generated_samples2.clone().detach().to(device=torch.device('cpu'))
        #     generated_samples2 = np.array(generated_samples2)
        #     generated_samples2=pd.DataFrame(data=generated_samples2,columns=index)
        #     generated_data=generated_data.append(generated_samples2)



        if distance < number_Features * 0.1:
            m = m + 1

        if (m>=10)and(epoch>=50):
            # print('在第{}轮收敛'.format(epoch))
            break
        # if (time.time() - start_time) > 10:
        #     train(df, first_time, args)
        #     return
    '''
    2022/2/22
    统计epoch
    '''

    # write_epoch(str(epoch), '../precision_baye_scode/')


    for k in range(len(dec_loss)):
        dec_loss[k] = dec_loss[k].detach()
        gen_loss[k] = gen_loss[k].detach()
    # Save plots of the training
    plt.plot(np.arange(0, epoch + 1, 1), np.array(dec_loss), label='Discriminator Loss')
    plt.plot(np.arange(0, epoch + 1, 1), np.array(gen_loss), label='Generator Loss')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Convergence for Net')
    plt.legend(loc='upper right')

    plt.savefig("GAN_Loss_DNN.png")
    plt.show()
    plt.close('all')


    # print(final_data)
    # print(final_data.shape[1])
    '''
     2022/1/13
     删选出偏移度小于2的10条数据

    '''
    generated_data = shuffle(generated_data).reset_index(drop=True)

    # 用于计算执行多少次循环
    count = 0
    while (len(final_data) < 100):
        latent_space_samples = torch.randn((200, args.number_features), device=torch.device(args.device))

        generated_samples = generator_mod(latent_space_samples)
        # m = discriminator_mod(real_samples.float())
        # m = np.array(m.clone().detach()).item()
        # write_DRData(str(m), '../precision_baye_scode/')
        # print('真实数据的结果为{}'.format(m))
        generated_samples2 = generated_samples.clone().detach().to(device=torch.device('cpu'))
        generated_samples = np.array(generated_samples2)
        generated_samples = pd.DataFrame(generated_samples, columns=index)
        samples, number_Features = generated_samples.shape
        generated_samples['range'] = 0
        generated_samples['dissimilarity_value'] = 0
        '''
                2022/2/12
                加入cos_distance距离

                '''
        generated_samples['cos_distance'] = 0
        '''
               2022/2/16
               加入欧式距离,总的距离度量

               '''
        generated_samples['Euclidean_distance'] = 0
        generated_samples['distance'] = 0
        '''
        2022/2/18
        加入判别器生成的结果
        '''

        generated_samples['discriminator_label'] = 0

        generated_samples['final_distance'] = 0
        '''
        2022/1/13
        将生成的数据转换到合法的范围之内【-1,1】中,并将其加入到偏移值’range‘列中
        '''
        for i in range(samples):
            for j in range(number_Features):
                if (datasets.iloc[0, j] == 0):
                    generated_samples['range'].iat[i] = generated_samples['range'].iat[i] + abs(
                        generated_samples.iloc[i, j])
                    generated_samples.iloc[i, j] = 0
                if (generated_samples.iloc[i, j] > 1):
                    generated_samples['range'].iat[i] = generated_samples['range'].iat[i] + abs(
                        (generated_samples.iloc[i, j] - 1) / datasets.iloc[0, j])
                    generated_samples.iloc[i, j] = 1
                if (generated_samples.iloc[i, j] < -1):
                    generated_samples['range'].iat[i] = generated_samples['range'].iat[i] + abs(
                        (1 - generated_samples.iloc[i, j]) / datasets.iloc[0, j])
                    generated_samples.iloc[i, j] = -1

        # print(generated_samples)
        generated_samples = generated_samples.reset_index(drop=True)
        '''
        2022/1/11
        根据取值范围将数据转换回去
        '''
        for indexs, row in generated_samples.iteritems():
            if indexs == 'range':
                continue
            if indexs == 'distance':
                continue
            if indexs == 'dissimilarity_value':
                continue
            if indexs == 'cos_distance':
                continue
            if indexs == 'Euclidean_distance':
                continue
            if indexs == 'discriminator_label':
                continue
            if indexs == 'final_distance':
                continue
            generated_samples[indexs] = sparkConfRangeDf.loc[indexs, 'min'] + ((generated_samples[indexs] - a) *
                                                                               (sparkConfRangeDf.loc[indexs, 'max'] -
                                                                                sparkConfRangeDf.loc[
                                                                                    indexs, 'min'])) / (b - a)
            try:
                if (df['spark.memory.offHeap.size'].item() == 0):
                    generated_samples['spark.memory.offHeap.size'] = 0
            except:
                print('')
            if sparkConfRangeDf.loc[indexs, 'pre'] == 1.0:
                generated_samples[indexs] = round(generated_samples[indexs])
            elif sparkConfRangeDf.loc[indexs, 'pre'] == 0.01:
                generated_samples[indexs] = round(generated_samples[indexs], 2)
        '''
        2022/1/13
        根据生成数据的要求要在[-0.25,0.25]中,继续计算偏移值。并将其加入到偏移值’range‘列中
        '''
        for i in range(samples):
            for j in range(number_Features):
                if (df.iloc[0, j] == 0):
                    generated_samples['range'].iat[i] = generated_samples['range'].iat[i] + abs(
                        generated_samples.iloc[i, j])
                    generated_samples.iloc[i, j] = 0
                if (generated_samples.iloc[i, j] > 1.25 * df.iloc[0, j]):
                    generated_samples['range'].iat[i] = generated_samples['range'].iat[i] + abs(
                        (generated_samples.iloc[i, j] - 1.25 * df.iloc[0, j]) / df.iloc[0, j])
                    generated_samples.iloc[i, j] = 1.25 * df.iloc[0, j]
                if (generated_samples.iloc[i, j] < 0.75 * df.iloc[0, j]):
                    generated_samples['range'].iat[i] = generated_samples['range'].iat[i] + abs(
                        (0.75 * df.iloc[0, j] - generated_samples.iloc[i, j]) / df.iloc[0, j])
                    generated_samples.iloc[i, j] = 0.75 * df.iloc[0, j]
                if (df.iloc[0, j] != 0):
                    generated_samples['dissimilarity_value'].iat[i] = generated_samples['dissimilarity_value'].iat[
                                                                          i] + abs(
                        (df.iloc[0, j] - generated_samples.iloc[i, j]) / df.iloc[0, j])

        '''
        2022/2/17
        ’disimilarity_value'应该计算平均每个维度变化的大小,distance重新度量
        '''
        generated_samples['dissimilarity_value'] = generated_samples['dissimilarity_value'] / number_Features
        # generated_samples['distance'] = generated_samples['distance'] / generated_samples['dissimilarity_value']

        generated_samples = generated_samples.sort_values('range').reset_index(drop=True)
        '''
        2022/2/16
        筛选出合法值,误差不能超过20%
        '''
        # 经过实验将range设置为3是比较合理的
        print('generated_samples\n' + str(generated_samples))
        generated_samples = generated_samples[generated_samples['range'] < 0.15 * number_Features]
        print('generated_samples.loc\n' + str(generated_samples.loc[:,['range']]))
        # if((time.time() - first_time)>60):
        #     final_data = final_data[final_data['range'] < 4]
        # elif(60>(time.time() - first_time)>40):
        #     final_data= final_data[final_data['range'] < 3]
        # else:
        #     final_data = final_data[final_data['range'] < 2]
        # print('---------------偏移值较小的数据-------------------')
        # print(final_data.loc[:, ['range', 'dissimilarity_value']])

        '''
        整体时间如果超过30s，则重新训练模型
        '''
        # if (time.time() - start_time) > 30:
        #     train(df, first_time, args)
        #     return
        '''
        2022/2/18
        将合法化的数据转换回去，用判别器判定
        '''
        label_data = dataset_to_below_1(args, generated_samples.iloc[:, :number_Features], sparkConfRangeDf)
        try:
            if (df['spark.memory.offHeap.size'].item() == 0):
                label_data['spark.memory.offHeap.size'] = 0
        except:
            print('')
        samples = generated_samples.shape[0]
        for i in range(samples):
            generated_samples['cos_distance'].iat[i] = cos_sim(label_data.iloc[i:i + 1, :df.shape[1]],
                                                               datasets)
            generated_samples['Euclidean_distance'].iat[i] = ed(label_data.iloc[i:i + 1, :df.shape[1]], datasets)
            generated_samples['distance'].iat[i] = generated_samples['cos_distance'].iat[i] / \
                                                   (generated_samples['Euclidean_distance'].iat[i] *
                                                    generated_samples['dissimilarity_value'].iat[i])
        '''
        2022/2/18
        判别器生成的标签
        '''

        label_data = torch.from_numpy(label_data.values)
        label_data = discriminator_mod(label_data.float())
        label_data = np.array(label_data.clone().detach().to(device=torch.device('cpu')))
        generated_samples['discriminator_label'] = label_data
        generated_samples['final_distance'] = generated_samples['discriminator_label'] * generated_samples['distance']
        final_data = final_data.append(generated_samples)

        # print('----------------第{}轮的原数据---------------------'.format(count))
        # print(generated_samples.loc[:,
        #       ['range', 'dissimilarity_value', 'cos_distance', 'Euclidean_distance', 'distance',
        #        'discriminator_label', 'final_distance']])

        count = count + 1
    for indexs, row in final_data.iteritems():
        try:
            if ((df['spark.memory.offHeap.size'].item() == 0) & (indexs == 'spark.memory.offHeap.size')):
                continue
        except:
            print('')
        try:
            if sparkConfRangeDf.loc[indexs, 'pre'] == 1.0:
                final_data[indexs] = round(final_data[indexs])
            elif sparkConfRangeDf.loc[indexs, 'pre'] == 0.01:
                final_data[indexs] = round(final_data[indexs], 2)
        except:
            final_data[indexs] = round(final_data[indexs], 5)

    # 按照dissimilarity_value排序后的数据
    # print("-----------------按照range排序----------")
    # final_data = final_data.sort_values('range').reset_index(drop=True)
    # # GetSplitData(final_data)
    # print(final_data.loc[:,
    #       ['range', 'dissimilarity_value', 'cos_distance', 'Euclidean_distance', 'distance', 'discriminator_label',
    #        'final_distance']])
    # print("-----------------按照dissimilarity_value排序----------")
    # final_data = final_data.sort_values('dissimilarity_value').reset_index(drop=True)
    # # GetSplitData(final_data)
    # print(final_data.loc[:,
    #       ['range', 'dissimilarity_value', 'cos_distance', 'Euclidean_distance', 'distance', 'discriminator_label',
    #        'final_distance']])
    #
    # print("-----------------按照cos_distance排序----------")
    # final_data = final_data.sort_values('cos_distance', ascending=False).reset_index(drop=True)
    # # GetSplitData(final_data)
    # print(final_data.loc[:,
    #       ['range', 'dissimilarity_value', 'cos_distance', 'Euclidean_distance', 'distance', 'discriminator_label',
    #        'final_distance']])
    # print("-----------------按照Euclidean_distance排序----------")
    # final_data = final_data.sort_values('Euclidean_distance').reset_index(drop=True)
    # # GetSplitData(final_data)
    # print(final_data.loc[:,
    #       ['range', 'dissimilarity_value', 'cos_distance', 'Euclidean_distance', 'distance', 'discriminator_label',
    #        'final_distance']])
    # print("-----------------按照distance排序----------")
    # final_data = final_data.sort_values('distance', ascending=False).reset_index(drop=True)
    # # GetSplitData(final_data)
    # print(final_data.loc[:,
    #       ['range', 'dissimilarity_value', 'cos_distance', 'Euclidean_distance', 'distance', 'discriminator_label',
    #        'final_distance']])
    # print("-----------------按照discriminator_label排序----------")
    # final_data = final_data.sort_values('discriminator_label', ascending=False).reset_index(drop=True)
    # # GetSplitData(final_data)
    # print(final_data.loc[:,
    #       ['range', 'dissimilarity_value', 'cos_distance', 'Euclidean_distance', 'distance', 'discriminator_label',
    #        'final_distance']])
    # print("-----------------按照final_distance排序----------")
    final_data = final_data.sort_values('final_distance', ascending=False).reset_index(drop=True)
    # GetSplitData(final_data)
    # print(final_data.loc[:,
    #       ['range', 'dissimilarity_value', 'cos_distance', 'Euclidean_distance', 'distance', 'discriminator_label',
    #        'final_distance']])
    # final_data.to_csv('../data_analysis/gan_sample.csv', index=None, mode='a')
    '''
    2022/1/20
    根据相似程度选择合适的样本点 0
    '''
    spilt = final_data.shape[0] // 3
    processing_data = final_data[0:3]
    processing_data = processing_data.append(final_data[spilt:spilt + 3])
    processing_data = processing_data.append(final_data[-3:])
    processing_data = processing_data.reset_index(drop=True)
    final_data = pd.DataFrame(processing_data, copy=True)
    for k in range(3):
        final_data[k * 3:k * 3 + 1] = processing_data[k:k + 1]
        final_data[k * 3 + 1:k * 3 + 2] = processing_data[k + 3:k + 3 + 1]
        final_data[k * 3 + 2:k * 3 + 3] = processing_data[8 - k:8 + 1 - k]

    final_data = final_data.reset_index(drop=True)
    # print(final_data.loc[:, ['distance']])
    final_data = final_data.drop('range', axis=1)
    final_data = final_data.drop('dissimilarity_value', axis=1)
    final_data = final_data.drop('cos_distance', axis=1)
    final_data = final_data.drop('Euclidean_distance', axis=1)
    final_data = final_data.drop('distance', axis=1)
    final_data = final_data.drop('discriminator_label', axis=1)
    final_data = final_data.drop('final_distance', axis=1)
    final_data = final_data.iloc[:10]

    # final_data.to_csv('../data_analysis/cosgan_sample.csv', index=None)
    '''
    将参数转为配置
    '''
    confDict = sparkConfRangeDf.to_dict('index')

    # 遍历csv每一组配置生成一个配置文件
    print('-----写入配置开始-----')
    for index, row in final_data.iterrows():
        # 打开配置文件模板
        fTemp = open(args.configTemp, 'r')
        # 复制模板，并追加配置
        fNew = open(args.csv_toconfig + '/config' + str(index), 'w')
        shutil.copyfileobj(fTemp, fNew, length=1024)
        try:
            for i, item in row.items():
                fNew.write(' ')
                fNew.write(i)
                fNew.write('\t')
                fNew.write(formatConf(confDict, i, item))
                fNew.write('\n')
        finally:
            fNew.close()
    print('-----写入配置完成-----')
    time_used = round((time.time() - first_time), 2)
    # write_time(str(time_used), '../precision_baye_scode/')
    print('sgan数据生成时间花费为：{}'.format(time_used))

    return  final_data


def GetSplitData(data):
    spilt = data.shape[0] // 3
    processing_data = data[0:3]
    processing_data = processing_data.append(data[spilt:spilt + 3])
    processing_data = processing_data.append(data[-3:])
    processing_data = processing_data.reset_index(drop=True)
    for k in range(3):
        data[k * 3:k * 3 + 1] = processing_data[k:k + 1]
        data[k * 3 + 1:k * 3 + 2] = processing_data[k + 3:k + 3 + 1]
        data[k * 3 + 2:k * 3 + 3] = processing_data[8 - k:8 + 1 - k]

    data = data.reset_index(drop=True)
    data = data[:6]
    data.to_csv('../data_analysis/snet_sample_config.csv', index=None, mode='a')
    data = data.drop('range', axis=1)
    data = data.drop('dissimilarity_value', axis=1)
    data = data.drop('cos_distance', axis=1)
    data = data.drop('Euclidean_distance', axis=1)
    data = data.drop('distance', axis=1)
    data = data.drop('discriminator_label', axis=1)
    data = data.drop('final_distance', axis=1)
    data.to_csv('../data_analysis/snet_config.csv', index=None, mode='a')
    return


# if __name__ == '__main__':
#     # 导入配置参数
#     args = parser.parse_args()
#     i=1
#     for i in range(100):
#         seed=random.randint(0,41)
#         print(seed)
#         df = pd.read_csv(args.sample_data, delimiter=',', encoding='utf-8', header=0)
#         df = df.iloc[seed:(seed+1), :-1]
#         first_time = time.time()
#         train(df, first_time, args)
#     '''
#     2022/2/20
#     直接输出配置
#     '''
    # args = parser.parse_args()
    # df = pd.read_csv(args.sample_data, delimiter=',', encoding='utf-8', header=0)
    # df = df.sort_values('runtime').reset_index(drop=True)
    # df = df.iloc[:1, :-1]
    # first_time = time.time()
    # train(df, first_time, args)
