#!/bin/bash
pcount=$#
if((pcount==0)); then
echo "args(benchmark) not enough";
exit;
fi
echo "------------------------micro/$1------------------------"
scp -r root@k8s-node02:/home/collect/csv/micro/$1 /home/collect/csv/before_clear/micro/node02/
scp -r root@k8s-node03:/home/collect/csv/micro/$1 /home/collect/csv/before_clear/micro/node03/
echo "------------------------os/$1------------------------"
scp -r root@k8s-node02:/home/collect/csv/os/$1 /home/collect/csv/before_clear/os/node02/
scp -r root@k8s-node03:/home/collect/csv/os/$1 /home/collect/csv/before_clear/os/node03/
# echo "------------------------container/$1------------------------"
# scp -r root@k8s-master01:/home/collect/csv/container/$1 /home/collect/csv/before_clear/container/
