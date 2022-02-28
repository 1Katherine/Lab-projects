import argparse
import torch
import os
# 获取当前文件路径
current_path = os.path.abspath(__file__)
father_path = os.path.abspath(os.path.dirname(os.path.dirname(current_path)) + os.path.sep + ".")

parser = argparse.ArgumentParser()

# basic parameters
parser.add_argument('--sample_data', type=str, default='../tearsort-20G---data/42terasort-20G-GAN-3+3.csv',
                    help='Data required to be mimced', required=False)
parser.add_argument('--config_range',type=str,default=father_path + '/Spark_conf_range_wordcount.xlsx',help='get config range and precision')
parser.add_argument('--csv_toconfig',type=str,default=father_path + '/config/',help='parameters to config to run ')
parser.add_argument('--configTemp',type=str,default=father_path + '/configTemp',help='config to run')

parser.add_argument('--result_dir', type=str, default='../GAN_result/', help='generator_data', required=False)
parser.add_argument('--result_model', type=str, default='../GAN_result/model/', help='model', required=False)
parser.add_argument('--cell_type', type=str, default='DNN', help='The type of cells :  lstm or gru',required=False)
parser.add_argument('--output_dir', type=str, default='../pretrained_models/', help='output directory', required=False)
parser.add_argument('--real_MIN_MAX', type=str, default='terasort-20G-MIN_MAX.csv', help="real data's MIN_MAX Value ")
parser.add_argument('--realData_normalization', type=str, default='terasort-20G-normalization.csv', help="normalizating the real data")
parser.add_argument('--type', type=str, default='terasort-20G', help='data_type', required=False)

# train parameters
parser.add_argument('--epochs', type=int, default=2000, help='epochs', required=False)
parser.add_argument('--batch_size', type=int, default=1, help='batch size used during training', required=False)
parser.add_argument('--number_features', type=int, default=84,
                    help='number of features in dataset has to include onehot encode columns(categories)',
                    required=False)
parser.add_argument('--hidden_size', type=int, default=128, help='number of hidden diemensions',required=False)
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate', required=False)
parser.add_argument('--beta1', type=float, default=0.5)
parser.add_argument('--beta2', type=float, default=0.999)
parser.add_argument('--num_layers', type=int, default=2, help='number of layers in the LSTM',required=False)
parser.add_argument('--d_size', type=int, default=3, help="one dimension to two dimension(picture)")
parser.add_argument('--device', type=str, default=torch.device("cpu"),
                    help='No CUDA cpu else cuda', required=False)


#generator data parameters
parser.add_argument('--generator_del', type=float, default=0.1, help='generator del %', required=False)
parser.add_argument('--real_generator', type=float, default=1, help='real and generator  %', required=False)

# Bayesian optimization parameters
parser.add_argument('--benchmark', type=str, help='benchmark type')
parser.add_argument('--initpoints', type=int, help='initpoints of bo， also by gan+rs generates')
parser.add_argument('--niters', type=int, help='iterarions of bo')
# parser.add_argument('--csv_toconfig',type=str,help='parameters to config to run ')

# 维护的参数-范围表
# args = parser.parse_args()
# args.benchmark = 'wordcount-100G'
# conf_range_table = father_path + "Spark_conf_range_"  + args.benchmark.split('-')[0] +  ".xlsx"
# print(conf_range_table)
# parser.add_argument('--config_range',type=str,default=conf_range_table,help='get config range and precision')
# print('configuration文件中的config_range = ' + str(args.config_range))
# print('configuration文件中的csv_toconfig = ' + str(args.csv_toconfig))
