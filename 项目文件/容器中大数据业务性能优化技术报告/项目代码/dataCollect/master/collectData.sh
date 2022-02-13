#!/bin/bash
pcount=$#
if((pcount < 3)); then
echo args not enough;
exit;
fi

echo "collect Micro architecture event"
for i in 02 03
do
        echo -------------------start k8s-node$i --------------
        ssh k8s-node$i "python /usr/local/home/zwr/micro_metric_collect.py --benchmark=$2 --jobid=$3" &
	ssh k8s-node$i "python /usr/local/home/zwr/os_metric_collect.py --benchmark=$2 --jobid=$3" &
done

echo -------------------start k8s-master01 --------------
# 获取开启脚本的时间戳
current=`date "+%Y-%m-%d %H:%M:%S"`
timeStamp=`date -d "$current" +%s`
startTimeStamp=$(((timeStamp*1000+10#`date "+%N"`/1000000)/1000))

python3 /usr/local/home/zwr/start_benchmark.py --path="$1"
echo $1
# 获取脚本结束的时间戳
current=`date "+%Y-%m-%d %H:%M:%S"`
timeStamp=`date -d "$current" +%s`
endTimeStamp=$(((timeStamp*1000+10#`date "+%N"`/1000000)/1000))

for i in 02 03
do
        echo -------------------stop k8s-node$i --------------
        ssh k8s-node$i "/usr/local/home/zwr/stop.sh"
done

# 容器层指标的采集
/usr/local/home/zwr/co_metric_collect.sh $2/comon-$3 $startTimeStamp $endTimeStamp

# 收集hibench运行时间
# tail -n 1 /usr/local/home/hibench/hibench/report/hibench.report >> /home/collect/runtime/$2-runtime.txt
