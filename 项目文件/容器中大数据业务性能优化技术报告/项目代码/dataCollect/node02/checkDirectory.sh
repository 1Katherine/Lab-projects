#!/bin/bash
pcount=$#
if((pcount == 0)); then
echo 'args(Directory) not enough';
exit;
fi

for i in "$@"
do
	echo --------- check $i ------------
	if [ ! -d $i ]; then
  		mkdir -p $i
	fi
done
