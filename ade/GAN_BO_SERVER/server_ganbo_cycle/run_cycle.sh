#!/bin/bash
#1 获取输入参数个数，如果没有参数，直接退出
pcount=$#
if((pcount<1)); then
echo "no arg(benchmark-size)";
exit;
fi

#gan+rs共生成8个初始样本提供给bo
initNumber=6
interationsNumber=6

path=$(pwd)
echo $path
# $1 = wordcount-100G
echo "=============== start $1 ===============" >> $path/direct_ganrs_Bayesian.log
echo $(date) >> $path/direct_ganrs_Bayesian.log
echo "=============== start $1 ===============" >> $path/direct_ganrs_Bayesian.log

startTime=$(date "+%m-%d-%H-%M")
mv  $path/config/$1 $path/config/$1-$startTime
mv $path/logs.json $path/config/$1-$startTime
mv $path/generationConf.csv $path/config/$1-$startTime
mv $path/target.png $path/config/$1-$startTime
mv $path/dataset.csv $path/config/$1-$startTime
mv $path/GAN* $path/config/$1-$startTime
mv $path/general_data.csv $path/config/$1-$startTime
mv $path/sgan_sample.csv $path/config/$1-$startTime


mkdir -p $path/config/$1

python3 $path/ganrs_Bayesian_Optimization_server_cycle.py --benchmark=$1 --initpoints=$initNumber --niters=$interationsNumber

finishTime=$(date "+%m-%d-%H-%M")
mv  $path/config/$1 $path/config/$1-$finishTime
mv $path/logs.json $path/config/$1-$finishTime
mv $path/generationConf.csv $path/config/$1-$finishTime
mv $path/target.png $path/config/$1-$finishTime
mv $path/dataset.csv $path/config/$1-$finishTime
mv $path/GAN* $path/config/$1-$finishTime
mv $path/general_data.csv $path/config/$1-$finishTime
mv $path/sgan_sample.csv $path/config/$1-$finishTime


echo "=============== finish $1 ===============" >> $path/direct_ganrs_Bayesian.log
echo $(date) >> $path/direct_ganrs_Bayesian.log
echo "=============== finish $1 ===============" >> $path/direct_ganrs_Bayesian.log
mv $path/direct_ganrs_Bayesian.log $path/direct_ganrs_Bayesian-$finishTime.log
