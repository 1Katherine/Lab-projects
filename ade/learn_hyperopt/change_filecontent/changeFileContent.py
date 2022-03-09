#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：learn_hyperopt 
@File ：changeFileContent.py
@Author ：Yang
@Date ：2022/3/1 11:02 
'''
import os

new_filename = 'shutDowonDetection_bak.py'
old_filename = 'shutDowonDetection.py'

stop_time = 50

# 打开旧文件
f = open(old_filename,'r',encoding='utf-8')
# 打开新文件
f_new = open(new_filename,'w',encoding='utf-8')
# 循环读取旧文件
for line in f:
    # 进行判断
    if line.find('maxRuntime') == 0:
        line_list = line.strip().split('=')
        line = line.replace(line_list[1], str(stop_time))
    # 如果不符合就正常的将文件中的内容读取并且输出到新文件中
    f_new.write(line)
f.close()
f_new.close()


# 获取当前文件路径
current_path = os.path.abspath(__file__)
# 获取当前文件的父目录,比如/usr/local/home/yyq/bo/rs_bo/rs_bo_newEI
father_path = os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".")


for file in os.listdir(father_path):
    if file == new_filename:
        try:
            os.rename(new_filename, old_filename)
        except FileExistsError:
            os.remove(old_filename)
            os.rename(new_filename, old_filename)
