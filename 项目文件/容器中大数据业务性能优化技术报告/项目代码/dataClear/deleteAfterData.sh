pcount=$#
if((pcount < 1)); then
echo "args(benchmark) not enough";
exit;
fi
rm -rf /home/collect/csv/after_clear/micro/$1
rm -rf /home/collect/csv/after_clear/os/$1
