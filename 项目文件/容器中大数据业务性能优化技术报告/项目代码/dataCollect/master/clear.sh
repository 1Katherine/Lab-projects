#!/bin/bash
pcount=$#
if((pcount < 2)); then
echo "args(benchmark, fileNumber) not enough";
exit;
fi


ssh k8s-node04 "/usr/local/home/zwr/aggregateCsv.sh $1;  /usr/local/home/zwr/clear.sh $1 $2"

