#!/bin/bash
pcount=$#
if((pcount < 2)); then
echo "args(FileNum,configPath) not enough";
exit;
fi

# terasort-20G
# 指定使用terasort运行的数据量大小为20G
rm -rf /usr/local/home/hibench/hibench/conf/workloads/micro/terasort.conf
cp /usr/local/home/zwr/hibench-conf/conf/terasort/terasort-20G.conf /usr/local/home/hibench/hibench/conf/workloads/micro/terasort.conf
# 使用指定配置文件跑terasort-20G
rm -rf /usr/local/home/hibench/hibench/conf/spark.conf
cp $2/config$1 /usr/local/home/hibench/hibench/conf/spark.conf

# 开启spark程序 
echo ================= config$1 =================
echo $(date)
python3 /usr/local/home/zwr/start_benchmark.py --path=/usr/local/home/hibench/hibench/bin/workloads/micro/terasort/spark/run.sh
# 清理pod的ip
/usr/local/home/zwr/stop.sh

# 采集指标
# /usr/local/home/zwr/oneBenchmarkCollect.sh /usr/local/home/hibench/hibench/bin/workloads/micro/wordcount/spark/run.sh wordcount-100G-ga $1 $1
