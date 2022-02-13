# 该类用于根据交互强度选择配置参数
import time


class ParameterSelect:
    def __init__(self, intensity_list, threshold,save_path):
        self.intensity_list = []
        self.intensity_list = intensity_list  # 存放交互强度的二维列表，intensity_list
                                              # 由parameters_events_interact.py中的
                                              #InteractCalculate类成员变量 intensity_list传入
                            #形如('L1-dcache-stores $ spark.io.compression.snappy.blockSize', 0.07152508169598973)
        self.threshold = threshold  # 筛选配置参数的标准，如果相邻两个交互强度比值大于该阈值，
        # 那么认为发生突变，往后的交互强度不再考虑

        self.selected_parameters = []  # 选择出的配置参数

        self.save_path=save_path

    def main(self):
        self.select()
        self.parameters_to_file()

    def select(self):
        for i in range(0, len(self.intensity_list)):
            if self.intensity_list[i][1] == 0 or abs(self.intensity_list[i][1] - 0.0) < pow(10, -3):
                break
            if i >= 1 and self.intensity_list[i - 1][1] / self.intensity_list[i][1] > self.threshold:
                break
            else:
                words = self.intensity_list[i][0].split()  # 按空格进行分割
                for word in words:
                    if word.startswith("spark"):  # 选出字符列表中的spark参数
                        self.selected_parameters.append(word) 


        self.selected_parameters=list(set(self.selected_parameters)) #进行去重

    def parameters_to_file(self):
        file=open(self.save_path+"selected_parameters.txt",mode="a+")
        file.write(str(time.localtime()))
        file.write("\n")
        file.write(str(len(self.selected_parameters)))
        file.write("\n")
        for parameter in self.selected_parameters:
            file.write(parameter)
            file.write("\n")

        file.close()



