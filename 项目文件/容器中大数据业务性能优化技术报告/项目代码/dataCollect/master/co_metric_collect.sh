#!/bin/bash
pcount=$#
if((pcount < 3)); then
echo "args(file_name, startTimeStamp, endTimeStamp) not enough";
exit;
fi
file_path=/home/collect/data/container

	file_name=$1
	startTimeStamp=$2
	endTimeStamp=$3

	mkdir -p ${file_path}/${file_name}	

	#获取运行时的pod名称
	curl "http://192.168.0.10:11000/api/v1/query_range?query=sum(kube_pod_container_status_running%7Bnamespace%3D%22default%22%7D)%20by%20(pod)&start=${startTimeStamp}&end=${endTimeStamp}&step=5" | grep -Po 'pod[" :]+\K[^"]+' >> podNames


	cat /usr/local/home/zwr/container_events_list | while read event
	do
		lines=$(echo $event | grep "kube")
        	if [ ! -n "${lines}" ]
		then
			touch ${file_path}/${file_name}/${event}.json
			cat podNames | while read line
			do
				curl "http://192.168.0.10:11000/api/v1/query_range?query=sum(${event}%7Bnamespace%3D%22default%22%2Cpod%3D~%22($line)%22%7D)%20by%20(pod)&start=${startTimeStamp}&end=${endTimeStamp}&step=1" >> ${file_path}/${file_name}/${event}.json

				echo "\n" >> ${file_path}/${file_name}/${event}.json
			done
		else
			if [ ! -n "$(echo ${lines} | grep "memory")" ]
                	then
 				touch ${file_path}/${file_name}/${lines}.json
                		cat podNames | while read line
                		do
                        		curl "http://192.168.0.10:11000/api/v1/query_range?query=sum(${lines%????}%7Bnamespace%3D%22default%22%2Cresource%3D%22cpu%22%2Cpod%3D~%22($line)%22%7D)%20by%20(pod)&start=${startTimeStamp}&end=${endTimeStamp}&step=1" >> ${file_path}/${file_name}/${lines}.json

                        		echo "\n" >> ${file_path}/${file_name}/${lines}.json
                		done
			else
				touch ${file_path}/${file_name}/${lines}.json
				cat podNames | while read line
                        	do
                                	curl "http://192.168.0.10:11000/api/v1/query_range?query=sum(${lines%???????}%7Bnamespace%3D%22default%22%2Cresource%3D%22memory%22%2Cpod%3D~%22($line)%22%7D)%20by%20(pod)&start=${startTimeStamp}&end=${endTimeStamp}&step=1" >> ${file_path}/${file_name}/${lines}.json

                                	echo "\n" >> ${file_path}/${file_name}/${lines}.json
                        	done
			fi
		fi
	done
	rm -rf podNames

echo "success"
