#!/bin/bash
pcount=$#
if((pcount < 3)); then
echo "args(benchamrk, start, end) not enough";
exit;
fi

for i in $(seq $2 1 $3)
do
	echo "============ $1/osmon-$i-k8s-node03 ============"
        python3 /usr/local/home/zwr/os_data_to_csv.py --benchmark=$1 --filename="osmon-$i-k8s-node03"
done
