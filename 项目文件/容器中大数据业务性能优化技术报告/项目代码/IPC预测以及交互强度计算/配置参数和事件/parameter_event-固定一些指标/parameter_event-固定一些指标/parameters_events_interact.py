import time

import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
import statsmodels.api as sfa

class InteractCalculate:

    def __init__(self, final_features, data, parameters_list, target,model,save_path,step_nums,num):
        self.sum_residual =0
        self.data = data  # 训练数据,微体系结构，os,容器三层事件，配置参数，ipc,执行时间。
        self.final_events =[]
        self.final_events=final_features  # 三层事件 列表
        
        self.save_path=save_path  #生成文件存储路径
        
        
        print("final events")
        print(len(self.final_events))
        file = open(self.save_path + "final_events.txt", mode="a+")
        for line in self.final_events:
            file.write(line)
            file.write("\n")
            
        self.parameters_list = [] # 配置参数
        self.parameters_list= parameters_list 
        self.target = target  # 预测目标，ipc或者执行时间


        self.num=num
        self.model=lgb.LGBMRegressor()
        self.model = model  # 预测模型

        self.events_parameters_list = [] #存放事件和配置参数
        
        for event in self.final_events:
            self.events_parameters_list.append(event)

        for parameter in self.parameters_list:
            self.events_parameters_list.append(parameter)
        
        self.events_parameters_pair_list=[]   #二维列表，存放事件和参数，第二维数据为一个事件和一个参数
        
        self.get_events_parameter_pair_list()        #将事件和参数 配对起来
        self.get_events_parameters_pair_residual(self.events_parameters_pair_list)   #构造事件-参数对 和 残差的字典   键为事件-参数对    值为残差

        self.residual_list={}     #记录 残差 的键值对，键为事件——参数对，值为 残差，排序后变为二维列表
        self.intensity_list = {}  #记录 交互强度 的键值对，键为事件——参数对，值为 交互强度，排序后变为二维列表
                                  #形如 ('L1-dcache-stores $ spark.io.compression.snappy.blockSize', 0.07152508169598973)
        print("events_parameters_list")
        print(len(self.events_parameters_pair_list))
        file = open(self.save_path + "pe_list.txt", mode="a+")
        for line in self.events_parameters_list:
            file.write(line)
            file.write("\n")
        file.close()
        

        self.step_nums=step_nums  #确定生成数据(用于做线性拟合)的大小  (step_nums+1)*(step_nums+1)即为生成数据的行数


    def main(self):
        i=0
        print("new_data")
        for event_pair in self.events_parameters_pair_list:
            self.make_data(event_pair=event_pair,step_nums=self.step_nums,index=i)
            i=i+1
        self.out_put()


    # 组合事件和配置参数
    def get_events_parameter_pair_list(self):

        for i in range(len(self.final_events)):
            for j in range(len(self.parameters_list)):
                temp_event_parameter_pair = []
                temp_event_parameter_pair.append(self.final_events[i])
                temp_event_parameter_pair.append(self.parameters_list[j])
                self.events_parameters_pair_list.append(temp_event_parameter_pair)

        print("events_parameter_pair_list")
        print(len(self.events_parameters_pair_list))
        file = open(self.save_path + "pe_pair_list.txt", mode="a+")
        file.write(str(len(self.events_parameters_pair_list)))
        for line in self.events_parameters_pair_list:
            file.write(str(line))
            file.write("\n")
        file.close()
       



    # 构造事件-参数对名称以及残差的对应关系 形如(msr/pperf/ $ spark.broadcast.blockSize, 0.03286879772327395)
    def get_events_parameters_pair_residual(self, pair_lists):
        combine_name_list = []
        for event_pair in pair_lists:
            event_pair_combineName = event_pair[0] + " $ " + event_pair[1]  # same_1
            combine_name_list.append(event_pair_combineName)

        values = [0 for i in range(len(combine_name_list))]

        self.residual_list = dict(zip(combine_name_list, values))
        print(type(self.residual_list))
        self.intensity_list = dict(zip(combine_name_list, values))
        #print(self.residual_list)

        # 生成新的训练数据,并使用lightgbm模型预测

    def make_data(self, event_pair, step_nums,index):
        # 先构建事件对里面的事件和参数的值
        # 新数据的行数
        rows = (step_nums + 1) * (step_nums + 1)
        array = np.zeros((rows, len(self.events_parameters_list)))
        # print(event_pair)
        new_data = pd.DataFrame(columns=self.events_parameters_list, data=array)

        evevt_1 = event_pair[0]
        evevt_1_min = self.data[evevt_1].min()
        evevt_1_max = self.data[evevt_1].max()
        evevt_1_step = (evevt_1_max - evevt_1_min) / step_nums

        evevt_2 = event_pair[1]
        evevt_2_min = self.data[evevt_2].min()
        evevt_2_max = self.data[evevt_2].max()
        evevt_2_step = (evevt_2_max - evevt_2_min) / step_nums

        block_nums = step_nums + 1
        for i in range(block_nums):
            for j in range(block_nums):
                cur_row = i * j  # 当前行数
                # 事件1是一整块取同一个值，一共(int(cur_row/block_nums))块，每一块的值不相同
                # 事件2是一整块取遍所有值，一共(int(cur_row/block_nums))块，每一块的值相同
                new_data.loc[cur_row, evevt_1] = (int(cur_row / block_nums)) * evevt_1_step + evevt_1_min
                new_data.loc[cur_row, evevt_2] = (int(cur_row) % (int(block_nums))) * evevt_2_step + evevt_2_min
        
        
     
        for feature in self.events_parameters_pair_list:
            if feature not in event_pair:
                new_data[feature] = self.data[feature].mean()
                
        if index%100==0:
            print("new_data")
            #new_data.to_csv("new_data"+str(index)+".csv")
        event_pair_combineName = event_pair[0] + " $ " + event_pair[1]  # same_1

        # 使用模型进行预测
        predictions = self.model.predict(new_data[self.events_parameters_list])

        self.residual_calculate(predictions=predictions, event_pair=event_pair,
                                event_pair_combineName=event_pair_combineName, data=new_data)

    def residual_calculate(self, predictions, event_pair, event_pair_combineName, data):

        events_data = data[event_pair]
        if self.linear_name.toLower()=="ols":
            result = sfa.OLS(predictions, events_data).fit().resid
        else:
            liner_model=LinearRegression()
            liner_model.fit(predictions,events_data)
            result=liner_model.residues

        temp_residual = result.resid  # 记录当前模型残差
        # print("residual")
        # print(temp_residual)
        # print(test_length)
        # print(type(temp_residual))

        self.residual_list[event_pair_combineName] = round(np.mean(pow(temp_residual, 2)) * 1000,2)

        # print(event_pair_combineName+":   "+str(temp_residual))

    def out_put(self):
        # 按残差大小排序
        # 字典排序后变成二维列表
        self.residual_list = sorted(self.residual_list.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
        print("residual_list")
        # print(self.residual_list)
        print(type(self.residual_list))
        print("残差-----------------")

        rate=1
        left_length = int(len(self.residual_list) / rate)
        # rate越大，剩下的越少
        self.residual_list = self.residual_list[0:left_length]
        
        file = open(self.save_path + "residual_" + str(self.num) + ".txt", mode="a+")
        file.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        file.write("\n")

        file.write(str(len(self.final_events)))
        file.write("\n")
        file.write(str(len(self.events_parameters_pair_list)))
        file.write("\n")
        for item in self.residual_list:
            file.write(str(item))
            file.write("\n")
        file.close()

        for name, value in self.residual_list:
            print(name + ": " + str(value))

        # 交互强度计算

        for name, value in self.residual_list:
            self.sum_residual += value  # 计算总残差

        for name, value in self.residual_list:
            self.intensity_list[name] = value / self.sum_residual  # 计算当前残差占总残差的比例

        self.intensity_list = sorted(self.intensity_list.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)

        self.intensity_list = self.intensity_list[0:left_length]

        print("交互强度---------------")
        plt.clf()
        event_name_plot = [component[0] for component in self.intensity_list[:10]]
        intensity_plot = [component[1] for component in self.intensity_list[:10]]
        sns.barplot(x=event_name_plot, y=intensity_plot)
        plt.xticks(rotation=270)
        plt.xlabel("event_parameter pair")
        plt.ylabel("intensity")
        plt.savefig(self.save_path + "intensity" + str(self.num) + ".png")
        plt.clf()

        for name, value in self.intensity_list:
            print(name + ": " + str(value))

        file = open(self.save_path + "intensity_" + str(self.num) + ".txt", mode="a+")
        file.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        file.write("\n")

        file.write("\n")
        file.write(str(len(self.final_events)))
        file.write("\n")
        file.write(str(len(self.events_parameters_pair_list)))
        file.write("\n")
        for item in self.intensity_list:
            file.write(str(item))
            file.write("\n")
        file.close()