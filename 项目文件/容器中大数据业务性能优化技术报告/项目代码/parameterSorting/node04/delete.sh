#!/bin/bash
pcount=$#
if((pcount < 1)); then
echo 'args(Directory) not enough';
exit;
fi

rm -rf /home/collect/csv/trainData/$1-sorting-parameters_runtime.csv /home/collect/csv/trainData/$1-important_parameter_runtime-parameters_runtime.csv
rm -rf /usr/local/home/zwr/hibench-spark-config/$1-sorting /usr/local/home/zwr/hibench-spark-config/$1-important_parameter_runtime
rm -rf /usr/local/home/zwr/log/$1-runningOutPut.log /usr/local/home/zwr/log/$1-parameterSorting.log
rm -rf /home/collect/trainResult/$1-sorting /home/collect/trainResult/$1-important_parameter_runtime
