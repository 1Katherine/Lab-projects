#!/bin/bash
pcount=$#
if((pcount < 4)); then
echo 'args(benchmarkPath, benchmarkName, start, end) not enough';
exit;
fi

for i in $(seq $3 1 $4)
do
        echo $(date)
	echo "************************* Begin $2 $i*************************"
	rm -rf /usr/local/home/hibench/hibench/conf/spark.conf
	cp /usr/local/home/zwr/hibench-spark-config/$2/config$i /usr/local/home/hibench/hibench/conf/spark.conf
	source /usr/local/home/tpcds/tpcds/tpcds/tpcds-env.sh
	/usr/local/home/zwr/collectData.sh "/usr/local/home/zwr/parameterSorting/tpcds-21G-sort.sh $i /usr/local/home/zwr/hibench-spark-config/$2 $2" $2 $i
	# /usr/local/home/zwr/collectData.sh $1 $2 $i
	echo "************************* Finish $2 $i*************************"
	echo $(date)
	echo "=========================================================================="
	sleep 3
	# tail -n 1 /usr/local/home/hibench/hibench/report/hibench.report >> /home/collect/runtime/$2-runtime.txt
done
# sleep 5
# tail -n `expr $4 - $3 + 1` /usr/local/home/hibench/hibench/report/hibench.report/ >> /home/collect/runtime/$2-runtime.txt
