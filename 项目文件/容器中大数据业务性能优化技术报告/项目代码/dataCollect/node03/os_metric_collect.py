#!/usr/bin/python3
import subprocess
import socket
import argparse

# Parse input options
parser = argparse.ArgumentParser(description="A System Metric Monitor.")
parser.add_argument('-i', '--interval', help='interval to read counters', type=int, default=1)
parser.add_argument('-c', '--command', help='following by a command or a program', nargs='+', default=[])
parser.add_argument('-j', '--jobid', help='slurm job id', type=int)
parser.add_argument('-b', '--benchmark', help='name of benchmark', type=str)
parser.add_argument('-o', '--options', help='options list using by dstat in file',
                    default='/usr/local/home/zwr/dstat_options.txt')
opts = parser.parse_args()

# The tool will read counters per intvl ms.
intvl = opts.interval

# The output file name
if opts.jobid is not None:
    fname = '/home/collect/data/os/%s/osmon-%d-%s.csv' % (opts.benchmark, opts.jobid, socket.gethostname())  # slurm job id, hostname
else:
    fname = '/home/collect/data/os/test/osmon-%s.csv' % (socket.gethostname())  # hostname


# This block use Linux command 'dstat' to monitor system metric
# filename = "options_list.txt"
with open(opts.options) as options_file:
    options = options_file.read().splitlines()

optionsWithoutSpace = []
for o in options:
    optionsWithoutSpace.append(o.strip())

options_str = ' '.join(optionsWithoutSpace)
# Construct dstat command
dstat_cmd = "dstat %s --output %s %d 1>/dev/null" % (options_str, fname, intvl)


p = subprocess.Popen(dstat_cmd, shell=True)
print("finish")

