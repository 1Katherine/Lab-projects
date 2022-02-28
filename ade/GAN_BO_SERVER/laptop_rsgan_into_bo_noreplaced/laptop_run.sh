#!/bin/bash
#1 获取输入参数个数，如果没有参数，直接退出
pcount=$#
if((pcount<1)); then
echo "no arg(benchmark-size)";
exit;
fi

#gan+rs共生成8个初始样本提供给bo
initNumber=3
#bo迭代搜索50-initNumber次
all_points=18
interationsNumber=$(($all_points - $initNumber))

path=$(pwd)
echo $path
# $1 = wordcount-100G



mkdir -p $path/config/$1
mkdir -p $path/SnetConfig/

python $path/ganrs_Bayesian_Optimization_ganinbo_noreplaced.py --benchmark=$1 --initpoints=$initNumber --niters=$interationsNumber --csv_toconfig=$path/SnetConfig/
