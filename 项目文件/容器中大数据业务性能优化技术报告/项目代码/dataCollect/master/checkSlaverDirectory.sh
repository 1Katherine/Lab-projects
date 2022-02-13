#!/bin/bash
pcount=$#
if((pcount < 2)); then
echo 'args (type, benchmarkName) not enough';
exit;
fi

microDirectory="/home/collect/$1/micro/$2"
osDirectory="/home/collect/$1/os/$2"

echo "***************** check directory *****************"
for j in 02 03
do
        echo -------------------check k8s-node$j --------------
        ssh k8s-node$j "/usr/local/home/zwr/checkDirectory.sh $microDirectory $osDirectory"
done
