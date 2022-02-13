import argparse
import warnings

import pandas as pd

import compare

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('-k', '--k8s_path', help='Path of k8sData')
parser.add_argument('-y', '--yarn_path', help='path of yarn')
parser.add_argument('-p', '--prefix', help='prefix of index')
parser.add_argument('-n', '--num', help='num of configurations')
parser.add_argument('-s', '--save_path', help='path for saving figures')
parser.add_argument('-o', '--option', help='choose which function to execute')
args = parser.parse_args()

k8s_path = args.k8s_path
yarn_path = args.yarn_path
prefix = args.prefix
num = args.num
save_path = args.save_path
option = args.option

k8s_data = pd.read_csv(k8s_path + prefix + str(1) + "-report.csv")
yarn_data = pd.read_csv(yarn_path + prefix + str(1) + "-report.csv")

yarn_features = yarn_data.columns
k8s_features = k8s_data.columns
intersection_features = list(set(yarn_features).intersection(set(k8s_features)))

# intersection_features=list(set(yarn_features).intersection(set(k8s_features)))
print(len(intersection_features))


compare_obj = compare.Compare_Yarn_K8s(k8s_path=k8s_path,yarn_path=yarn_path,init_features=intersection_features,save_path=save_path,
                                       num=num,option=option,prefix=prefix)

if option=="line_plot":
    compare_obj.line_plot()
elif option=="sub_plot":
    compare_obj.config_subplot()
elif option=="calculate":
    compare_obj.calculate()
else:
    compare_obj.dtw_calculate()
