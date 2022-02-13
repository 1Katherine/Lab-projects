pcount=$#
if((pcount < 1)); then
echo "args(benchmark) not enough";
exit;
fi
rm -rf /home/collect/csv/before_clear/micro/node02/$1
rm -rf /home/collect/csv/before_clear/micro/node03/$1
rm -rf /home/collect/csv/before_clear/os/node02/$1
rm -rf /home/collect/csv/before_clear/os/node03/$1
