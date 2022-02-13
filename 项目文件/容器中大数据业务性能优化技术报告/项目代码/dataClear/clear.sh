#!/bin/bash
pcount=$#
if((pcount < 2)); then
echo "args(benchmark, fileNumber) not enough";
exit;
fi

microDirectory="/home/collect/csv/after_clear/micro/$1"
osDirectory="/home/collect/csv/after_clear/os/$1"
coDirectory="/home/collect/csv/after_clear/container/$1"
intDirectory="/home/collect/csv/after_clear/integration/$1"

# 为各层合并的数据建立目录
# /usr/local/home/zwr/checkDirectory.sh $intDirectory
/usr/local/home/zwr/checkDirectory.sh $microDirectory
/usr/local/home/zwr/checkDirectory.sh $osDirectory
# /usr/local/home/zwr/checkDirectory.sh $coDirectory

# 清洗各层的数据
python3 /usr/local/home/zwr/clear_micro.py --benchmark=$1 --number=$2
python3 /usr/local/home/zwr/clear_os.py --benchmark=$1 --number=$2
# python3 /usr/local/home/zwr/clear_container.py --benchmark=$1 --number=$2

# 分别合并各层的数据
# python3 /usr/local/home/zwr/merge.py --benchmark=$1 --type=micro
# python3 /usr/local/home/zwr/merge.py --benchmark=$1 --type=os
# python3 /usr/local/home/zwr/merge.py --benchmark=$1 --type=container

# 按配置文件编号将各层的数据整合在一起
# python3 /usr/local/home/zwr/integrate.py --benchmark=$1 --number=$2

# 将整合好的数据拼成一张大的表格
# python3 /usr/local/home/zwr/merge.py --benchmark=$1 --type=integration
echo "------------------- clear successful -------------------"
