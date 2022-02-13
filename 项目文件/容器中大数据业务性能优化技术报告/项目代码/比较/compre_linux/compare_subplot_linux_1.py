import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
from dtaidistance import dtw

parser=argparse.ArgumentParser()
parser.add_argument('-k','--k8s_path',help='Path of k8sData')
parser.add_argument('-y','--yarn_path',help='path of yarn')
parser.add_argument('-p','--prefix',help='prefix of index')
parser.add_argument('-n','--num',help='num of configurations')
parser.add_argument('-s','--save_path',help='path for saving figures')
args=parser.parse_args()

k8s_path=args.k8s_path
yarn_path=args.yarn_path
prefix=args.prefix
num=args.num
save_path=args.save_path

k8s_data =pd.read_csv(k8s_path+prefix+str(1)+"-report.csv")
yarn_data=pd.read_csv(yarn_path+prefix+str(1)+"-report.csv")

#print(k8s_data.describe())

#print(yarn_data.describe())


#画图，把各个指标的最大值，均值，分位数，众数，中位数，画出来

#训练，得出重要指标及重要性差异

#画图，得出变化趋势，比如说折线图，一张图画两个折线

yarn_features=yarn_data.columns
k8s_features=k8s_data.columns
union_features=list(set(yarn_features).union(set(k8s_features)))

print(len(union_features))




#用于读取数据
def getData(path_k8s,path_yarn,index,prefix):
    path_yarn = path_yarn +prefix+str(index)+"-report.csv"
    path_k8s = path_k8s + prefix+str(index)+"-report.csv"
    #print(path_yarn+"\n")
    #print(path_k8s+"\n")
    yarn_data = pd.read_csv(path_yarn)
    k8s_data = pd.read_csv(path_k8s)

    # 特征取交集
    yarn_features = yarn_data.columns
    k8s_features = k8s_data.columns
    intersect_features = list(set(yarn_features).intersection(set(k8s_features)))
    print(len(intersect_features))

    yarn_data = yarn_data[intersect_features]
    k8s_data = k8s_data[intersect_features]

    return yarn_data,k8s_data





def process_data(k8s_data,yarn_data):
    k8s_category = ['k8s' for i in range(len(k8s_data))]
    yarn_category = ['yarn' for i in range(len(yarn_data))]

    k8s_data.insert(0, 'category', k8s_category)
    yarn_data.insert(0, 'category', yarn_category)


    concat_data=pd.concat([yarn_data,k8s_data])

    return concat_data





def line_plot(yarn_data,k8s_data,intersect_features,path):
    for feature in intersect_features:
        sns.lineplot(data=yarn_data[feature], label="yarn")
        sns.lineplot(data=k8s_data[feature], label="k8s")
        plt.xlabel(feature)
        feature = feature.replace("/", "_")
        print(feature)
        plt.savefig(path+"\\"+feature + ".png")
        plt.clf()
        # plt.show()

def box_plot(yarn_data,k8s_data,intersect_features,path):
    for feature in intersect_features:
        sns.boxplot(data=yarn_data[feature],color="blue")
        sns.boxplot(data=k8s_data[feature],color="red")
        plt.xlabel(feature)
        feature = feature.replace("/", "_")
        print(feature)
        plt.savefig(path+"\\"+feature+".png")
        plt.clf()




def data_compare(k8s_data,yarn_data,plot_data,intersect_features,path):
    for feature in intersect_features:
        plt.subplot(2,1,1)
        sns.boxplot(x="category", y=feature, data=plot_data)
        plt.xlabel(feature)
        print(feature)

        plt.subplot(2,1,2)
        sns.lineplot(data=yarn_data[feature], label="yarn",color="blue")
        sns.lineplot(data=k8s_data[feature], label="k8s",color="orange")

        feature = feature.replace("/", "_")
        plt.savefig(path+"\\"+feature + ".png")
        plt.clf()


def calculate(path_k8s,path_yarn,num,prefix,intersect_features):
        min_mean_features=[]
        max_mean_features=[]

        key_list=intersect_features
        value_list=[0 for i in range(len(key_list))]
        minor_features_dict = dict(zip( key_list ,value_list))
        major_features_dict = dict(zip(key_list,value_list))
        for index in range(1, num + 1):
            yarn_data, k8s_data = getData(path_k8s=path_k8s, path_yarn=path_yarn, index=index,prefix=prefix)
            #concat_data = process_data(yarn_data=yarn_data, k8s_data=k8s_data)
            intersect_features=yarn_data.columns

            comparsion_data=pd.DataFrame(index=["max","min","median","mean","std","num"],columns=intersect_features)


            count=0
            for feature in intersect_features:
                #最大值差异
                max_Yarn=yarn_data[feature].max()
                #print("max_yarn",max_Yarn)

                max_K8s=k8s_data[feature].max()
                #print("max_k8s", max_K8s)
                try:
                    comparsion_data.loc["max",feature] = (max_K8s - max_Yarn) / max_Yarn
                except ZeroDivisionError:
                    print("Zero")
                    comparsion_data.loc["max",feature] = np.nan


                #最小值差异
                min_Yarn = yarn_data[feature].min()
                min_K8s = k8s_data[feature].min()
                try:
                    comparsion_data.loc["min",feature]=(min_K8s-min_Yarn)/min_Yarn
                except ZeroDivisionError:
                    print("Zero")
                    comparsion_data.loc["min",feature]=np.nan

                #中位数差异
                median_Yarn = yarn_data[feature].median()
                median_K8s = k8s_data[feature].median()
                try:
                    comparsion_data.loc["median",feature] = (median_K8s - median_Yarn) / median_Yarn
                except ZeroDivisionError:
                    print("Zero")
                    comparsion_data.loc["median",feature] = np.nan

                #平均值差异
                mean_Yarn = yarn_data[feature].mean()
                mean_K8s = k8s_data[feature].mean()
                try:
                    comparsion_data.loc["mean",feature] = (mean_K8s - mean_Yarn) / mean_Yarn
                    if abs(comparsion_data.loc["mean",feature])<0.1:
                        minor_features_dict[feature]+=1
                    if abs(minor_features_dict[feature])>num*0.8:
                        min_mean_features.append(feature)


                    if abs(comparsion_data.loc["mean",feature])>0.3:
                        major_features_dict[feature]+=1
                    if abs(major_features_dict[feature])>num*0.7:
                        max_mean_features.append(feature)

                except ZeroDivisionError :
                    print("Zero")
                    comparsion_data.loc["mean",feature] = np.nan
                except KeyError:
                    print("no such feature "+feature)


                #标准差差异
                std_Yarn = yarn_data[feature].std()
                std_K8s = k8s_data[feature].std()
                try:
                    comparsion_data.loc["std",feature] = (std_K8s - std_Yarn) / std_Yarn
                except ZeroDivisionError:
                    print("Zero")
                    comparsion_data.loc["std",feature] = np.nan

                #数量差异
                num_Yarn = len(yarn_data[feature])
                num_K8s = len(k8s_data[feature])
                try:
                    comparsion_data.loc["num", feature] = (num_K8s - num_Yarn) / num_Yarn
                except ZeroDivisionError:
                    print("Zero")
                    comparsion_data.loc["num", feature] = np.nan

            #comparsion_data.to_csv("C:\\Users\douglF\Desktop\ADE\\train_data\compare_data\\values\micro\stat\wordcount-100g\\"+str(index)+".csv")

        print("max_mean_features")
        print(set(max_mean_features))

        print("min_mean_features")
        print(set(min_mean_features))

            # 最小值差异


def config_subplot(path_k8s,path_yarn,num,intersect_features,save_path):
    #subplot的行和列
    row=np.sqrt(num)
    if row>round(row):
        col=row+1
    else:
        col=row
    #每个特征画一个图，每个图由所有配置的子图构成
    for feature in intersect_features:

        for index in range(1,num+1):
            yarn_data,k8s_data=getData(path_k8s=path_k8s,path_yarn=path_yarn,index=index)
            concat_data=process_data(yarn_data=yarn_data,k8s_data=k8s_data)

            cur_row = index / col
            cur_col = index - cur_row * col
            # 当前子图位置
            plt.subplot(row,col,index)
            try:
                sns.boxplot(data=concat_data,x="category",y=concat_data[feature])
                plt.xlabel(" ")
                plt.ylabel(" ")
            except BaseException:
                print("No such feature"+feature+"\n")

            feature=feature.replace("/","_")
            plt.savefig(save_path+feature+".png")
        plt.clf()


def dtw_calculate(path_k8s,path_yarn,num,intersect_features,prefix):
    min_dtw_features = []
    max_dtw_features = []

    key_list = intersect_features
    value_list = [0 for i in range(len(key_list))]
    dtw_features_dict = dict(zip(key_list, value_list))

    for index in range(1, num + 1):
        yarn_data, k8s_data = getData(path_k8s=path_k8s, path_yarn=path_yarn, index=index, prefix=prefix)
        # concat_data = process_data(yarn_data=yarn_data, k8s_data=k8s_data)
        intersect_features = yarn_data.columns
        for feature in intersect_features:
            print(feature)
            try:
                distance=dtw.distance(k8s_data[feature],yarn_data[feature])
                print(distance)
                dtw_features_dict[feature]+=distance
            except BaseException:
                print("Error")


    dtw_features_dict=sorted(dtw_features_dict.items() ,key=lambda kv:(kv[1], kv[0]))

    #features_=dtw_features_dict.keys()

    print("min_dtw")
    print(dtw_features_dict[0:10])
    
    print("\n,max_dtw")
    length=len(dtw_features_dict)
    print(dtw_features_dict[length-10:length])
    

    
    #dist_=dtw_features_dict.values()
    #print(features_[0:3])
    #print("\n")
    #print(dist_[0:3])

    #feature_length=len(features_)

    #print(features_[ feature_length-3 : feature_length ])
    #print("\n")
    #print(dist_[feature_length-3:feature_length])








#concat_data=process_data(k8s_data=k8s_data,yarn_data=yarn_data)
#concat_data.to_csv("concat.csv")

#print(yarn_data.columns)

#print(k8s_data.columns)
#box_plot(plot_data=concat_data)
#line_plot(k8s_data=k8s_data,yarn_data=yarn_data)

#data_compare(k8s_data=k8s_data,yarn_data=k8s_data,plot_data=concat_data)

#box_plot(plot_data=concat_data,path="C:\\Users\douglF\Desktop\ADE\\train_data\compare_data\\values\os\\box_plot")
#line_plot(k8s_data=k8s_data,yarn_data=yarn_data,path="C:\\Users\douglF\Desktop\ADE\\train_data\compare_data\\values\os\lines_plot")




config_subplot(path_k8s=path_k8s,
               path_yarn=path_yarn,
          num=num ,intersect_features=intersect_features)
"""
calculate(path_k8s=path_k8s,
          path_yarn=path_yarn,
          num=num,prefix=prefix,intersect_features=intersect_features)

#line_plot(k8s_data=k8s_data,yarn_data=yarn_data,intersect_features=intersect_features,path="C:\\Users\douglF\Desktop\ADE\\train_data\compare_data\\values\os\lines_plot\wordcount-100g-comparsion1")
dtw_calculate(path_k8s=k8s_path,path_yarn=yarn_path,
          num=int(num),prefix=prefix,intersect_features=intersect_features)"""