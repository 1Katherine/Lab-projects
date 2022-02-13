#! /bin/bash

for (( i = 0; i < 5; i++ )); do
  mkdir /usr/local/home/train_data/python_file/interact/event_event/files/files$i
  python3 main_3.py -n lgb -s /usr/local/home/train_data/python_file/interact/event_event/files/files$i/ -f /usr/local/home/zwr/wordcount-100G.csv  -step 1 -left 70 -t instructions


done