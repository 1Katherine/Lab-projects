#2021 6/25  新创建dataFrame,前两列是交互对事件，后面是其他事件。
# 交互对事件按步长从最小值到最大值生成，其他事件取平均值。
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import datetime
import time


import statsmodels.api as sfa


"""
parser=argparse.ArgumentParser()
parser.add_argument('-f','--filePath',help='Path of data')
parser.add_argument('-e','--eventPath',help='Path of event')
args=parser.parse_args()
"""


class Interaction_calculate:
    def __init__(self,final_features,data,num,model,step_nums,save_path):

        #只包含重要事件和ipc的dataFrame，由feature_choose训练完毕后传入
        self.data=data
        #重要事件列表
        self.final_features=final_features

        self.save_path=str(save_path) #生成文件存储路径
        #选几个事件
        self.num=num
        
        self.final_features=self.final_features[0:num]

        #feature_choose训练完毕后传入的训练模型
        self.model=model


        #构建新数据时，将最小值和最大值之间划出的份数
        self.step_nums=step_nums
        if self.step_nums<=0:
            print("Error","step_nums must be positive int")
            return

        #标签名
        self.label=self.data.columns[-1]
        #标签数据
        self.target_data = self.data[self.label]


        #一维列表，存放事件
        self.events_list=[]

        #二维列表，存放事件对。第二维数据为长度为2的列表，存放一组事件对。
        self.events_pair_list=[]

        #记录事件对的残差
        self.residual_list={}
        #记录交互强度
        self.intensity_list={}

        #残差总和
        self.sum_residual=0

        #初始化成员变量

        self.get_events_pair_list()
        self.get_events_pair_residual()



        print("residual-------------------")
        #print(self.residual_list)

    def main(self):

        for event_pair in self.events_pair_list:

            self.make_data(event_pair=event_pair,step_nums=self.step_nums)
        self.out_put()




    #构造事件对名称以及残差的对应关系，键值对，键为事件对名称，值为残差
    def get_events_pair_residual(self):
        combine_name_list=[]
        for event_pair in self.events_pair_list:
            event_pair_combineName =event_pair[0] + " $ " + event_pair[1]   #same_1
            combine_name_list.append(event_pair_combineName)

        values=[0 for i in range(len(combine_name_list))]

        self.residual_list=dict(zip(combine_name_list,values))
        print(type(self.residual_list))
        self.intensity_list=dict(zip(combine_name_list,values))

    #组合事件形成事件对
    def get_events_pair_list(self):
        for i in range(len(self.final_features)):
            #print(i)
            for j in range(i+1,len(self.final_features)):
                temp_event_pair=[]
                temp_event_pair.append(self.final_features[i])
                temp_event_pair.append(self.final_features[j])
                self.events_pair_list.append(temp_event_pair)

        #print(len(self.events_pair_list))


    #生成新的训练数据,并使用lightgbm模型预测
    def make_data(self,event_pair,step_nums):
        #先构建事件对里面的两个事件的值
        # 新数据的行数
        rows = (step_nums + 1) * (step_nums + 1)
        array=np.zeros((rows,len(self.final_features)))
        #print(event_pair)
        new_data=pd.DataFrame(columns=self.final_features,data=array)

        evevt_1=event_pair[0]
        evevt_1_min=self.data[evevt_1].min()
        evevt_1_max=self.data[evevt_1].max()
        evevt_1_step=(evevt_1_max-evevt_1_min)/step_nums

        evevt_2=event_pair[1]
        evevt_2_min = self.data[evevt_2].min()
        evevt_2_max = self.data[evevt_2].max()
        evevt_2_step = (evevt_2_max - evevt_2_min) / step_nums

        block_nums=step_nums+1
        for i in range(block_nums):
            for j in range(block_nums):
                cur_row=i*j #当前行数
                #事件1是一整块取同一个值，一共(int(cur_row/block_nums))块，每一块的值不相同
                #事件2是一整块取遍所有值，一共(int(cur_row/block_nums))块，每一块的值相同
                new_data.loc[cur_row,evevt_1]=(int(cur_row/block_nums))*evevt_1_step+evevt_1_min
                new_data.loc[cur_row,evevt_2]=(int(cur_row)%(int(block_nums)))*evevt_2_step+evevt_2_min

        #其他特征取平均值
        for feature in self.final_features:
            if feature not in event_pair:
                new_data[feature]=self.data[feature].mean()

        event_pair_combineName = event_pair[0] + " $ " + event_pair[1]  # same_1

        #使用feature_choose的最终模型进行预测
        predictions=self.model.predict(new_data)



        self.residual_calculate(predictions=predictions,event_pair=event_pair,event_pair_combineName=event_pair_combineName,data=new_data)



    def residual_calculate(self,predictions,event_pair,event_pair_combineName,data):

        features_data=data[self.final_features]
        result=sfa.OLS(predictions,features_data).fit()
        #print("result")
        #print(result)


        temp_residual =result.resid  #记录当前模型残差
        #print("residual")
        #print(temp_residual)
        # print(test_length)
        #print(type(temp_residual))



        self.residual_list[event_pair_combineName]=np.mean(pow(temp_residual,2))*1000

        #print(event_pair_combineName+":   "+str(temp_residual))



    def out_put(self):
        #按残差大小排序
        #字典排序后变成二维列表
        self.residual_list=sorted(self.residual_list.items(),key=lambda kv:(kv[1], kv[0]),reverse=True)
        print("residual_list")
        #print(self.residual_list)
        print(type(self.residual_list))
        print("残差-----------------")
   
        left_length=int(len(self.residual_list)/5)
        #取残差最高的前20%
        self.residual_list=self.residual_list[0:left_length]
        
        for name,value in self.residual_list:
            print(name+": "+str(value))




        # 交互强度计算

        for name, value in self.residual_list:
            self.sum_residual+=value   #计算总残差

        for name, value in self.residual_list:
            self.intensity_list[ name ]=value/self.sum_residual  #计算当前残差占总残差的比例

        self.intensity_list=sorted(self.intensity_list.items(), key=lambda kv:(kv[1], kv[0]),reverse=True)
    
        self.intensity_list=self.intensity_list[0:left_length]


        print("交互强度---------------")
        plt.clf()
        event_name_plot=[ component[0] for component in self.intensity_list[:10]]
        intensity_plot= [ component[1] for component in self.intensity_list[:10]]
        sns.barplot(x=event_name_plot,y=intensity_plot)
        plt.xticks(rotation=270)
        plt.xlabel("事件对")
        plt.ylabel("交互强度")
        plt.savefig(self.save_path+"intensity"+str(self.num)+".png")
        plt.clf()

        for name, value in self.intensity_list:
            print(name + ": " + str(value))

        file = open(self.save_path+"intensity_"+str(self.num)+".txt", mode="a+")
        file.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        file.write("\n")

        file.write("\n")
        file.write(str(len(self.final_features)))
        file.write("\n")
        file.write(str(len(self.events_pair_list)))
        file.write("\n")
        for item in self.intensity_list:
            file.write(str(item))
            file.write("\n")

