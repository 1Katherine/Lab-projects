#!/bin/bash
for i in 4040 4041 4042 4043
do
rm -rf /usr/local/home/zwr/errorDetection/monitor.html
python3 /usr/local/home/zwr/errorDetection/shutDowonDetection.py -n=$i
done