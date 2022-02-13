# coding=UTF-8
#构造
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
import os
import time_predict


parser=argparse.ArgumentParser()
parser.add_argument('-f','--filePath',help='Path of trainData')
parser.add_argument('-n','--name',help='name of algorithm')
parser.add_argument('-s','--save_path',help='path for saving files')
parser.add_argument('-t','--target',help='prediction target')


args=parser.parse_args()

filepath=args.filePath
name=args.name
save_path=args.save_path
target=str(args.target)


def get_data(file_path, name):
    data = pd.read_csv(file_path)
    all_columns = data.columns

    column_length = len(all_columns)

    print("\n")
    print(name + " 特征个数: ")
    print(str(column_length - 1))

    print(name + " 行数: ")
    print(str(len(data)))

    # 存放特征
    features_list = []
    for feature in all_columns:
        if feature != target:
            features_list.append(feature)
    return data, features_list


data, features_list = get_data(file_path=filepath, name="parameters")

model_construct_Obj= time_predict.Model_construct(name=name,save_path=save_path,target=target,data=data,features=features_list)
model_construct_Obj.main()


