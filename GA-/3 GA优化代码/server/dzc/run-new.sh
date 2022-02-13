#!/bin/bash
#1 获取输入参数个数，如果没有参数，直接退出
pcount=$#
if((pcount<1)); then
echo "no arg(benchmark-size)";
exit;
fi


chrom_group=30
interationsNumber=10


path=$(pwd)
echo $path
# $1 = wordcount-100G
echo "=============== start $1 ===============" >> $path/ACS_cluster.log
echo $(date) >> $path/ACS_cluster.log
echo "=============== start $1 ===============" >> $path/ACS_cluster.log

startTime=$(date "+%m-%d-%H-%M")
mv  $path/config/$1 $path/config/$1-$startTime
mv $path/generationBestConf.csv $path/config/$1-$startTime
mv $path/target.png $path/config/$1-$startTime
mv $path/gaConfs.csv $path/config/$1-$startTime
mv $path/all_history_y.csv $path/config/$1-$startTime


mkdir -p $path/config/$1

python3 $path/ACS_cluster.py  --benchmark=$1 --group=$chrom_group --niters=$interationsNumber

finishTime=$(date "+%m-%d-%H-%M")
mv  $path/config/$1 $path/config/$1-$finishTime
mv $path/generationBestConf.csv $path/config/$1-$finishTime
mv $path/target.png $path/config/$1-$finishTime
mv $path/gaConfs.csv $path/config/$1-$startTime
mv $path/all_history_y.csv $path/config/$1-$startTime

echo "=============== finish $1 ===============" >> $path/ACS_cluster.log
echo $(date) >> $path/ACS_cluster.log
echo "=============== finish $1 ===============" >> $path/ACS_cluster.log
mv $path/ACS_cluster.log $path/ACS_cluster-$finishTime.log
