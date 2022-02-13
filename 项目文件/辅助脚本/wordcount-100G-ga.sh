#!/bin/bash
pcount=$#
if((pcount < 2)); then
echo "args(FileNum,configPath) not enough";
exit;
fi

# wordcount-100G
# 指定使用wordcount运行的数据量大小为100G
rm -rf /usr/local/home/hibench/hibench/conf/workloads/micro/wordcount.conf
cp /usr/local/home/zwr/hibench-conf/conf/wordcount/wordcount-100G.conf /usr/local/home/hibench/hibench/conf/workloads/micro/wordcount.conf
# 使用指定配置文件跑wordcount-100G
rm -rf /usr/local/home/hibench/hibench/conf/spark.conf
cp $2/config$1 /usr/local/home/hibench/hibench/conf/spark.conf

# 开启spark程序 
echo ================= config$1 =================
echo $(date)
python3 /usr/local/home/zwr/start_benchmark.py --path=/usr/local/home/hibench/hibench/bin/workloads/micro/wordcount/spark/run.sh
# 清理pod的ip
/usr/local/home/zwr/stop.sh

# 采集指标
# /usr/local/home/zwr/oneBenchmarkCollect.sh /usr/local/home/hibench/hibench/bin/workloads/micro/wordcount/spark/run.sh wordcount-100G-ga $1 $1
