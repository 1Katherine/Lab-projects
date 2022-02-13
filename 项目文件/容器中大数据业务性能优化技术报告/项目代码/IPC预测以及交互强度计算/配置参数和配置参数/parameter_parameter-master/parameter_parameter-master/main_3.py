#读取最初的总数据(包含micro,os,container)，对三层分别进行特征选择

import parameter_choose  #用于特征选择的类
import micro_interact_new   #用于计算交互强度的类

#coding=UTF-8
import argparse
import datetime
import random
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import joblib
from sklearn.preprocessing import StandardScaler, normalize


parser=argparse.ArgumentParser()
parser.add_argument('-f','--filePath',help='Path of trainData')
parser.add_argument('-n','--name',help='name of algorithm')
parser.add_argument('-s','--save_path',help='path for saving files')
parser.add_argument('-t','--target',help='prediction target')
parser.add_argument('-step','--step_nums',help='the num of parameters droped each time')
parser.add_argument('-left','--left_nums',help='the num of parameters needed to be selected')

args=parser.parse_args()

filepath=args.filePath
name=args.name
save_path=args.save_path
target=str(args.target)
step=int(args.step_nums)
left_num=int(args.left_nums)

fig=plt.figure()
fig.tight_layout()



def get_data(file_path,name):
    data=pd.read_csv(file_path)
    all_columns=data.columns

    column_length=len(all_columns)

    print("\n")
    print(name+" 特征个数: ")
    print(str(column_length-1))

    print(name+" 行数: ")
    print(str(len(data)))

    #存放特征
    features_list= []
    for feature in all_columns:
        if feature!="runtime":
            features_list.append(feature)
    return data,features_list


#总数据
data,features_list=get_data(file_path=filepath,name="m-o-c")



total_choose=parameter_choose.Choose(name=name,features=features_list,step=step,prefix="m_o_c",data=data,save_path=save_path,target=target,left_num=left_num)
total_choose.main()




sum_importance=0
for i in range(len(total_choose.final_features)):
    sum_importance+=total_choose.final_importance[i]

cur_sum=0
left_features=[]
for i in range(len(total_choose.final_importance)):
    cur_sum+=total_choose.final_importance[i]
    left_features.append(total_choose.final_features[i])
    if cur_sum>=sum_importance*0.80:
        break

file=open(save_path+"selected_parameters.txt",mode="a+")
file.write(str(len(left_features)))
file.write("\n")
for feature in left_features:
    file.write(feature)
    file.write("\n")
file.close()









"""

interact_obj=micro_interact_new.Interaction_calculate(data=data,final_features=total_choose.final_features,
                                                      num=len(total_choose.final_features),
                                                      model=total_choose.model,
                                                      step_nums=20,save_path=save_path)

interact_obj.main()

"""
