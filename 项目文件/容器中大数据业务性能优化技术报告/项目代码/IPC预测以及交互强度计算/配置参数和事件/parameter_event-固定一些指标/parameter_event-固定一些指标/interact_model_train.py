import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split


class ParameterEventsTrain:
    def __init__(self,final_features,data,parameters_list,target,save_path):
        self.data=data  #训练数据,微体系结构，os,容器三层事件，配置参数，ipc,执行时间。
        self.final_events=[] #三层事件 列表
        self.final_events=final_features 
        
        self.parameters_list=[] #配置参数
        self.parameters_list=parameters_list  
        
        self.target=target #预测目标，ipc或者执行时间

        self.model=lgb.LGBMRegressor()  #训练模型

        self.events_parameters_list=[]  #存放事件和参数
        print("self.data.columns")
        print(len(self.data.columns))

        self.save_path=save_path
    
    def main(self):
        self.make_features()
        self.train()
        
    #将事件和配置参数都放进特征列表中
    def make_features(self):
        for event in self.final_events:
            self.events_parameters_list.append(event)
        for parameter in self.parameters_list:
            self.events_parameters_list.append(parameter)
        print("make_features")
        print(len(self.events_parameters_list))
        file = open(self.save_path+"pe2_list.txt", mode="a+")
        for line in self.events_parameters_list:
            file.write(line)
            file.write("\n")

    # 切分数据做标准化
    def process_data(self, features):
        # 切分数据集,测试集占0.25

        features_data = self.data[features]
        target_data = self.data[self.target]

        x_train, x_test, y_train, y_test = train_test_split(features_data, target_data, test_size=0.25, random_state=22)

        # 做标准化`
        # transfer = StandardScaler()
        # x_train = transfer.fit_transform(x_train)
        # x_test = transfer.transform(x_test)
        return x_train, x_test, y_train, y_test

    def train(self):
        x_train,x_test,y_train,y_test=self.process_data(features=self.events_parameters_list)
        model = lgb.LGBMRegressor()
        model.fit(x_train,y_train)
        features=[]
        features=self.events_parameters_list
        # 记录特征重要性
        features_importance =[]
        features_importance=model.feature_importances_
        # 将特征和特征重要性拼到一起 格式如右 [('RM', 0.49359385750858875), ('LSTAT', 0.3256110013950264)]
        features_with_importance = list(zip(features, features_importance))
        y_predict=model.predict(x_test)
        
        # 根据特征重要性进行排序，component[1]为重要性
        # 按降序排序
        features_with_importance = sorted(features_with_importance, key=lambda component: component[1], reverse=True)
        
        record_features =[x[0] for x in features_with_importance]
        record_importance=[x[1] for x in features_with_importance]
        
        file=open(self.save_path+"pe_features.txt",mode="a+")
        for i in range(len(record_features)):
            file.write(record_features[i])
            file.write(": ")
            file.write(str(record_importance[i]))
            file.write("\n")
        file.close()
        
        # 计算误差
        error_percentage=self.error_calculate(y_predict=y_predict,y_test=y_test)
        

        print("三层事件加特征 误差")
        print(error_percentage)


        features_data = self.data[self.events_parameters_list]
        target_data = self.data[self.target]

        self.model.fit(features_data,target_data)

    def error_calculate(self, y_predict, y_test):
        y_test = y_test.tolist()
        test_length = len(y_test)
        error_percentage = 0
        # print(test_length)

        for i in range(0, test_length):
            # print(y_predict[i])
            error_percentage = error_percentage + (abs(y_test[i] - y_predict[i]) / y_test[i])

        # 所有误差取平均值

        error_percentage = error_percentage / test_length

        return error_percentage
