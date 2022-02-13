import matplotlib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
from dtaidistance import dtw
import warnings
from matplotlib import cm
from matplotlib.pyplot import MultipleLocator
# x_major_locator=MultipleLocator(35)
import matplotlib.ticker as ticker
import category_map

plt.tight_layout()

category_dict = category_map.index_category

matplotlib.rcParams.update({'font.size': 15})


class Compare_Yarn_K8s:
    def __init__(self, init_features, k8s_path, yarn_path, save_path, prefix, num, option):
        self.intersection_features = []
        self.intersection_features = init_features

        self.k8s_path = k8s_path
        self.yarn_path = yarn_path

        self.save_path = save_path  # 生成数据存储路径
        self.prefix = prefix  # 文件名前缀

        self.num = int(num)  # 采用的文件个数
        self.option = option  # 选择使用哪个方法

    # 用于读取数据
    def getData(self, k8s_path, yarn_path, index, prefix):
        cur_yarn_path = yarn_path + prefix + str(index) + "-report.csv"
        cur_k8s_path = k8s_path + prefix + str(index) + "-report.csv"
        # print(yarn_path+"\n")
        # print(k8s_path+"\n")
        yarn_data = pd.read_csv(cur_yarn_path)
        k8s_data = pd.read_csv(cur_k8s_path)

        # 特征取并集
        yarn_features = yarn_data.columns
        k8s_features = k8s_data.columns
        intersection_features = list(set(yarn_features).intersection(set(k8s_features)))
        # print(len(self.intersection_features))

        yarn_data = yarn_data[intersection_features]
        k8s_data = k8s_data[intersection_features]

        return yarn_data, k8s_data, intersection_features

    def process_data(self, k8s_data, yarn_data):  # 将K8s和yarn数据拼到一起并添加类别，用于画盒图
        k8s_category = ['k8s' for i in range(len(k8s_data))]
        yarn_category = ['yarn' for i in range(len(yarn_data))]

        k8s_data.insert(0, 'category', k8s_category)
        yarn_data.insert(0, 'category', yarn_category)

        concat_data = pd.concat([yarn_data, k8s_data])

        return concat_data

    # 画折线图，同一数据大小，同一配置的情况下，同一指标的分布情况
    def line_plot(self):
        for index in range(1, self.num + 1):
            yarn_data, k8s_data, features_cur = self.getData(k8s_path=self.k8s_path, yarn_path=self.yarn_path,
                                                             index=index,
                                                             prefix=self.prefix)
            non_meaning = 0
            # concat_data = process_data(yarn_data=yarn_data, k8s_data=k8s_data)
            for feature in self.intersection_features:
                if feature in features_cur:
                    sns.lineplot(data=yarn_data[feature], label="yarn")
                    sns.lineplot(data=k8s_data[feature], label="k8s")
                    plt.xlabel(feature)
                    feature = feature.replace("/", "_")
                    # print(feature)
                    plt.savefig(self.save_path + "line_plots" + "/" + feature + "_" + str(index) + ".jpg")
                    plt.clf()
                else:
                    # print("No such feature: " + feature + "\n")
                    non_meaning += 1

    # 做各种信息统计，并生成csv文件。统计量包含中位数，平均值，最小值，最大值等
    def calculate(self):
        global mean_diff_dict
        min_mean_features = []
        max_mean_features = []

        mean_count_dict = dict(zip([feature for feature in self.intersection_features],
                                   [0 for i in
                                    range(len(self.intersection_features))]))  # 同一指标 k8s环境下平均值比  yarn环境下大的  个数
        median_count_dict = dict(zip([feature for feature in self.intersection_features],
                                     [0 for i in
                                      range(len(self.intersection_features))]))  # 同一指标 k8s环境下中位数比  yarn环境下大的  个数

        key_list = self.intersection_features
        value_list = [0 for i in range(len(key_list))]
        minor_features_dict = dict(zip(key_list, value_list))
        major_features_dict = dict(zip(key_list, value_list))

        def draw_bar(diff_dict,name,x_tick_size,y_tick_size,textHeight,figsize_X,figsize_Y,dpi,title,title_size):
            plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
            plt.rcParams['axes.unicode_minus'] = False

            # x轴是特征名，Y轴是(mean_k8s)/(mean_k8s+yarn_k8s)
            diff_list = sorted(diff_dict.items(), key=lambda kv: (kv[1], kv[0]))

            features_list = []
            values_list = []
            # 加入类别，按类别再排序一次
            for i in range(len(diff_list)):
                feature = diff_list[i][0]
                category_tuple = (category_dict[feature],)
                diff_list[i] += category_tuple

            diff_list = sorted(diff_list, key=lambda kv: (kv[2], kv[1]))

            category_list = []

            for item in diff_list:
                features_list.append(item[0])
                values_list.append(item[1])
                category_list.append(item[2])

            def autolable(rects):
                for rect in rects:
                    height = rect.get_height()

                    if "count" in name:
                        plt.text(rect.get_x() + rect.get_width() / 2.0 - 0.3, height + 0.012, '%.0f' % height, size=textHeight)
                    else:
                        plt.text(rect.get_x() + rect.get_width() / 2.0 - 0.3, height + 0.012, '%.2f' % height,
                                 size=textHeight)

            """
            norm = plt.Normalize(0, 1)
            #设置颜色
            norm_values = norm(values_list)
            map_vir = cm.get_cmap(name='inferno')
            colors = map_vir(norm_values)

            fig = plt.figure()  # 调用figure创建一个绘图对象
            plt.subplot(111)
            """

            colors = []
            colors_choice ={True:"blue",False:"red"}
            flag=True
            last_category=1
            for category_item in category_list:
                if last_category==category_item:
                    colors.append(cur_color)
                else:
                    flag=not flag
                    cur_color = colors_choice[flag]
                    last_category=category_item
                    colors.append(cur_color)

            plt.tight_layout()
            plt.figure(figsize=(figsize_X, figsize_Y), dpi=dpi)

            plt.title(title, fontdict={'family': 'Times New Roman', 'size': title_size})
            plt.yticks(np.arange(0, 1.05, 0.1), fontproperties='Times New Roman', size=y_tick_size)
            plt.xticks(rotation=270, size=x_tick_size)



            ax = plt.bar(features_list, values_list, color=colors, edgecolor='black', width=0.5)  # edgecolor边框颜色

            """
            sm = cm.ScalarMappable(cmap=map_vir, norm=norm)  # norm设置最大最小值
            sm.set_array([])
            tick_locator = ticker.MaxNLocator(nbins=4)  # colorbar上的刻度值个数

            cb1=plt.colorbar(sm)
            cb1.locator = tick_locator
            cb1.update_ticks()
            """

            autolable(ax)

            if "count" in name:
                plt.savefig(self.save_path + "bar_plots" + "/" + "mean" + "/" + "diff" + "/" + name + ".jpg")
            else:
                plt.savefig(self.save_path + "bar_plots" + "/" + name + "/" + "diff" + "/" + str(index) + ".jpg")

            plt.clf()


        # 遍历同一数据大小下，os或micro的所有配置运行结果
        for index in range(1, self.num + 1):
            yarn_data, k8s_data, features_cur = self.getData(k8s_path=self.k8s_path, yarn_path=self.yarn_path,
                                                             index=index,
                                                             prefix=self.prefix)
            # concat_data = process_data(yarn_data=yarn_data, k8s_data=k8s_data)
            self.intersection_features = yarn_data.columns

            comparsion_data = pd.DataFrame(index=["max", "min", "median", "mean", "std", "num"],
                                           columns=self.intersection_features)

            # 计算公式 k8s_mean/(k8s_mean+yarn_mean)
            mean_diff_dict = dict(zip([feature for feature in self.intersection_features],
                                      [0 for i in
                                       range(len(self.intersection_features))]))

            # 计算公式 k8s_median/(k8s_median+yarn_median)
            median_diff_dict = dict(zip([feature for feature in self.intersection_features],
                                        [0 for i in
                                         range(len(self.intersection_features))]))

            count = 0
            for feature in self.intersection_features:
                non_sense = 0
                # 最大值差异
                max_Yarn = yarn_data[feature].max()
                # print("max_yarn",max_Yarn)

                max_K8s = k8s_data[feature].max()
                # print("max_k8s", max_K8s)
                try:
                    comparsion_data.loc["max", feature] = (max_K8s - max_Yarn) / max_Yarn
                except ZeroDivisionError:
                    # print("Zero")
                    comparsion_data.loc["max", feature] = np.nan

                # 最小值差异
                min_Yarn = yarn_data[feature].min()
                min_K8s = k8s_data[feature].min()
                try:
                    comparsion_data.loc["min", feature] = (min_K8s - min_Yarn) / min_Yarn
                except ZeroDivisionError:
                    # print("Zero")
                    comparsion_data.loc["min", feature] = np.nan

                # 中位数差异
                median_Yarn = yarn_data[feature].median()
                median_K8s = k8s_data[feature].median()
                try:
                    comparsion_data.loc["median", feature] = (median_K8s - median_Yarn) / median_Yarn
                    if comparsion_data.loc["median", feature] > 0:
                        median_count_dict[feature] += 1
                    median_diff_dict[feature] = median_K8s / (median_Yarn + median_K8s)
                except Exception:
                    # print("Zero")
                    comparsion_data.loc["median", feature] = np.nan

                # 平均值差异
                mean_Yarn = yarn_data[feature].mean()
                mean_K8s = k8s_data[feature].mean()
                try:
                    comparsion_data.loc["mean", feature] = (mean_K8s - mean_Yarn) / mean_Yarn
                    if comparsion_data.loc["mean", feature] > 0:
                        mean_count_dict[feature] += 1
                    if abs(comparsion_data.loc["mean", feature]) < 0.1:
                        minor_features_dict[feature] += 1
                    if abs(minor_features_dict[feature]) > self.num * 0.8:
                        min_mean_features.append(feature)

                    if abs(comparsion_data.loc["mean", feature]) > 0.3:
                        major_features_dict[feature] += 1
                    if abs(major_features_dict[feature]) > self.num * 0.7:
                        max_mean_features.append(feature)

                    mean_diff_dict[feature] = (mean_K8s) / (mean_Yarn + mean_K8s)


                except Exception:
                    # print("Zero")
                    comparsion_data.loc["mean", feature] = np.nan
                except KeyError:
                    # print("no such feature "+feature)
                    non_sense += 1

                # 标准差差异
                std_Yarn = yarn_data[feature].std()
                std_K8s = k8s_data[feature].std()
                try:
                    comparsion_data.loc["std", feature] = (std_K8s - std_Yarn) / std_Yarn
                except ZeroDivisionError:
                    # print("Zero")
                    comparsion_data.loc["std", feature] = np.nan

                # 数量差异
                num_Yarn = len(yarn_data[feature])
                num_K8s = len(k8s_data[feature])
                try:
                    comparsion_data.loc["num", feature] = (num_K8s - num_Yarn) / num_Yarn
                except ZeroDivisionError:
                    # print("Zero")
                    comparsion_data.loc["num", feature] = np.nan

            # comparsion_data.to_csv(self.save_path + "csv" + "/" + str(index) + ".csv")

            draw_bar( diff_dict=mean_diff_dict, name="mean",x_tick_size=80,y_tick_size=120,
                      textHeight=50,figsize_X=100,figsize_Y=75,dpi=100,title="mean_k8s/(mean_k8s+mean_yarn)",title_size=120)
            # draw_diff_bar(title="median_k8s/(median_k8s+median_yarn)", diff_dict=median_diff_dict, path="median")

        def write_count(count_dict, name):
            file = open(self.save_path + name + ".txt", mode="a+")
            file.write(name + ": ")
            file.write("\n")
            for item in count_dict.items():
                file.write(str(item))
                file.write("\n")
            file.close()

        # write_count(mean_count_dict,"mean_count")
        # write_count(median_count_dict, "median_count")

        # 记录该指标有多少次是k8s比yarn大

        draw_bar(mean_count_dict, name="mean_count",x_tick_size=80,y_tick_size=120,
                      textHeight=50,figsize_X=100,figsize_Y=75,dpi=100,
                       title="The number of times feature in k8s is bigger than that in yarn: compared by mean_value  "
                 ,title_size=120)

        #draw_count_bar(median_count_dict, name="median_count",x_tick_size=80,y_tick_size=120,
        # textHeight=50,figsize_X=100,figsize_Y=75,dpi=100,
        # title="The number of times feature in k8s is bigger than that in yarn: compared by median_value  ",title_size=120)

        # self.get_Mean_Meadian_info()

    # 画盒图，反应数据的分布情况
    def config_subplot(self):
        # subplot的行和列
        row = np.sqrt(self.num)
        if row > round(row):
            col = row + 1
        else:
            col = row
        # 每个特征画一个图，每个图由所有配置的子图构成
        for feature in self.intersection_features:
            non_meaning = 0
            for index in range(1, self.num + 1):

                yarn_data, k8s_data, features_cur = self.getData(k8s_path=self.k8s_path, yarn_path=self.yarn_path,
                                                                 index=index,
                                                                 prefix=self.prefix)
                concat_data = self.process_data(yarn_data=yarn_data, k8s_data=k8s_data)

                cur_row = index / col
                cur_col = index - cur_row * col
                # 当前子图位置
                plt.subplot(row, col, index)
                if feature in features_cur:
                    sns.boxplot(data=concat_data, x="category", y=feature)
                    plt.xlabel(" ")
                    plt.ylabel(" ")
                else:
                    # print("No such feature: " + feature + "\n")
                    non_meaning += 1

                new_feature = feature.replace("/", "_")
                plt.savefig(self.save_path + "sub_plots" + "/" + new_feature + ".jpg")
            plt.clf()

    # 被calculate调用，将中位数和平均值的差异用图显示出来，并记录哪些配置是K8s大，哪些是yarn大,k8s大记1，否则记0
    def get_Mean_Meadian_info(self):
        non_meaning = 0

        def draw_feature_count_bar(new_feature, count_data, path):
            x_data = [i for i in range(1, self.num + 1)],

            plt.bar(x_data, count_data)
            plt.savefig(self.save_path + "bar_plots" + "/" + path + "/" + new_feature + ".jpg")
            plt.yticks()
            plt.figure(figsize=(60, 40), dpi=220)
            plt.clf()

        for feature in self.intersection_features:
            feature_median_info = [0 for i in range(1, self.num + 1)]
            feature_mean_info = [0 for i in range(1, self.num + 1)]
            for index in range(1, self.num + 1):
                data = pd.read_csv(self.save_path + "csv" + "/" + str(index) + ".csv")
                # print(data)
                try:
                    feature_median_info[index - 1] = data.loc[2, feature]
                    feature_mean_info[index - 1] = data.loc[3, feature]
                except Exception:
                    non_meaning += 1
            new_feature = feature.replace("/", "_")

            draw_feature_count_bar(new_feature=new_feature, count_data=feature_mean_info, path="mean")
            draw_feature_count_bar(new_feature=new_feature, count_data=feature_median_info, path="median")

    # 计算dtw距离，反应该指标的差异
    def dtw_calculate(self):
        print(self.k8s_path)
        print(self.yarn_path)
        print("\n")
        min_dtw_features = []
        max_dtw_features = []

        key_list = self.intersection_features
        value_list = [0 for i in range(len(key_list))]
        dtw_features_dict = dict(zip(key_list, value_list))

        for index in range(1, self.num + 1):
            print(index)
            yarn_data, k8s_data, features_cur = self.getData(k8s_path=self.k8s_path, yarn_path=self.yarn_path,
                                                             index=index,
                                                             prefix=self.prefix)
            # concat_data = process_data(yarn_data=yarn_data, k8s_data=k8s_data)
            non_meaning = 0
            for feature in features_cur:
                # print(feature)
                try:
                    distance = dtw.distance(k8s_data[feature], yarn_data[feature])
                    # print(distance)
                    dtw_features_dict[feature] += distance
                except BaseException:
                    non_meaning += 1

        dtw_features_dict = sorted(dtw_features_dict.items(), key=lambda kv: (kv[1], kv[0]))

        # features_=dtw_features_dict.keys()

        file = open(str(self.save_path) + "dtw.txt", mode="a+")

        print("min_dtw")
        print(dtw_features_dict[0:15])

        file.write("min dtw-----------------\n")
        for item in dtw_features_dict[0:15]:
            file.write(str(item))
            file.write("\n")

        print("\n,max_dtw")
        length = len(dtw_features_dict)

        file.write("\n")
        file.write("max dtw ----------------\n")
        max_dtw_features_dict = dtw_features_dict[length - 15:length]
        print(dtw_features_dict[length - 15:length])

        for item in reversed(max_dtw_features_dict):
            file.write(str(item))
            file.write("\n")
