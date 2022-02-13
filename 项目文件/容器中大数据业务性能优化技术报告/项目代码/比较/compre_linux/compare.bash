#!/bin/bash

arr=("micro" "os")
for i in 1 ; do
    let size=75
    #mkdir /usr/local/home/train_data/python_file/compare/wordcount-${size}G
    for type in  ${arr[@]}

    do
      #mkdir /usr/local/home/train_data/python_file/compare/wordcount-${size}G/${type}
	  #mkdir /usr/local/home/train_data/python_file/compare/wordcount-${size}G/${type}/plots
	  #mkdir /usr/local/home/train_data/python_file/compare/wordcount-${size}G/${type}/csv
	  
	  python3 /usr/local/home/train_data/python_file/compare/compare_subplot_linux.py     							-k /home/collect/csv/after_clear/${type}/wordcount-${size}G-comparison-3/ 			                          -y /home/collect/csv/after_clear/${type}/yarn-wordcount-${size}G-comparison-3/   								-p ${type}-    -n 20  -s /usr/local/home/train_data/python_file/compare/wordcount-${size}G/${type}/   -o subplot
    done

done