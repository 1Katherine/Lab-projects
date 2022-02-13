#!/bin/bash
pcount=$#
if((pcount < 1)); then
echo "args(Directory) not enough";
exit;
fi

cp -r /usr/local/home/zwr/hibench-spark-config/$1 /usr/local/home/zwr/hibench-spark-config/yarn-$1

echo ----------------- K8s-configuration to Yarn-configuration ---------------------

for((i=1;i<=2000;i++)); do

sed -i 's/spark.executor.cores/hibench.yarn.executor.cores/g' /usr/local/home/zwr/hibench-spark-config/yarn-$1/config$i
sed -i 's/spark.executor.instances/hibench.yarn.executor.num/g' /usr/local/home/zwr/hibench-spark-config/yarn-$1/config$i
sed -i '4c hibench.spark.master     yarn-client' /usr/local/home/zwr/hibench-spark-config/yarn-$1/config$i
sed -i '7c # spark.kubernetes.container.image     192.168.0.40/library/spark:v2.4.4' /usr/local/home/zwr/hibench-spark-config/yarn-$1/config$i
sed -i '5,6d' /usr/local/home/zwr/hibench-spark-config/yarn-$1/config$i

done
