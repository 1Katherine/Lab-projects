#!/bin/bash
pcount=$#
if((pcount < 3)); then
echo "args(FileNum, ConfigPath, storePath) not enough";
exit;
fi

# 使用指定配置文件跑tpcds-21G
rm -rf /usr/local/home/spark/spark/conf/spark-defaults.conf
cp $2/config$1 /usr/local/home/spark/spark/conf/spark-defaults.conf


# 开启spark程序 
echo ================= config$1 =================
echo $(date)
source /usr/local/home/tpcds/tpcds/tpcds/tpcds-env.sh
python3 /usr/local/home/tpcds/tpcds/tpcds/tpcdsMulti.py 4 $3 $1 True $QUERY_SQL_DIR $QUERY_RESULT_DIR $TPCDS_DBNAME $HDFS_NAME $TPCDS_SCALE_FACTOR
# 清理pod的ip
# /usr/local/home/zwr/stop.sh
