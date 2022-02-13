#读取最初的总数据(包含micro,os,container)，对三层分别进行特征选择

import feature_choose  #用于特征选择的类
import parameters_events_interact  #用于计算交互强度的类
import parameter_select  #用于选出重要参数的类
import interact_model_train
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
parser.add_argument('-t','--target',help='predict target')
parser.add_argument('-p','--pe_path',help='parameter and events data')

parser.add_argument('-step', '--steps', help='the num of parameters be droped each train')


args=parser.parse_args()

filepath=args.filePath
name=args.name
save_path=args.save_path
target=args.target
pe_path=args.pe_path
step = int(args.steps)



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
        if feature != "instructions":
            features_list.append(feature)
    return data,features_list


#总数据
data,features_list=get_data(file_path=filepath,name="m-o-c")
print("events")

#参数和事件 数据
pe_data,pe_features=get_data(file_path=pe_path,name="pe")
print('\n')
parameters=[]
parameters=pe_features[0:86]
print("parameters")
print(len(parameters))
print(parameters)

total_choose=feature_choose.Choose(name=name,features=features_list,step=step,prefix="m_o_c",data=data,save_path=save_path)

total_choose.main()

final_features=[]
final_features=total_choose.final_features
sort_final_features=[]
sort_final_features=total_choose.sort_final_features

print("events")
print(len(total_choose.final_features))
print(total_choose.final_features)



model_train=interact_model_train.ParameterEventsTrain(data=pe_data,final_features=sort_final_features,
target=target,parameters_list=parameters,save_path=save_path)

model_train.main()




length=len(total_choose.final_features)+len(parameters)    #参数和事件数量
print("length")
print(length)



interact_obj=parameters_events_interact.InteractCalculate(final_features=sort_final_features,
                                                          parameters_list=parameters,data=pe_data,
                                                          save_path=save_path,step_nums=30,num=length,
                                                          model=model_train.model,target=target)


interact_obj.main()

parameter_select_obj=parameter_select.ParameterSelect(intensity_list=interact_obj.intensity_list,threshold=pow(10,5),save_path=save_path)

parameter_select_obj.main()



