#!/bin/bash
pcount=$#
if((pcount < 3)); then
echo "args(benchmark, start, end) not enough";
exit;
fi

# os和micro层数据格式转换
echo "***************** dat file to csv file *****************"
for j in 02 03
do
	echo ------------------- FormatChange k8s-node$j  --------------
	ssh k8s-node$j "/usr/local/home/zwr/microFormatChang.sh $1 $2 $3; /usr/local/home/zwr/osFormatChang.sh $1 $2 $3"
done

# 容器层数据格式转换
# for i in $(seq $2 1 $3)
# do
# echo "======================== $1/comon-$i ========================"
# python3 /home/yyq/containerDataFirstDict.py --dpath=/home/collect/data/container/$1/comon-$i/
# done
echo "***************** $1 Format changed successfully *****************"
