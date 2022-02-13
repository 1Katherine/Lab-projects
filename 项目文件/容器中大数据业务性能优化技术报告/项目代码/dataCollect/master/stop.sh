#!/bin/bash
for i in 02 03
do
        echo -------------------stop k8s-node$i --------------
        ssh k8s-node$i "/usr/local/home/zwr/stop.sh"
done
