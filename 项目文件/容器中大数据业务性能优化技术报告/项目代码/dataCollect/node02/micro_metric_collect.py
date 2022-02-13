#!/usr/bin/python3
# NOTE!!! This monitor script only works for Intel Xeon E5/E7 series
# For scalable platforms (starts from skylake), no support yet. 
# To use this script, one needs to modify the 'cps' variable according to the CPU
import subprocess
import socket
import argparse
import os
from datetime import datetime


# Colorful output
class pfmon_color:
    red = "\x1b[31m"
    green = "\x1b[32m"
    yellow = "\x1b[33m"
    blue = "\x1b[34m"
    magenta = "\x1b[35m"
    cyan = "\x1b[36m"
    reset = "\x1b[0m"


# Parse input options
parser = argparse.ArgumentParser(description="A PMU Monitor.")
parser.add_argument('-i', '--interval', help='interval to read counters', type=int, default=95)
parser.add_argument('-p', '--pid', help='attach to a process by pid', type=int)
parser.add_argument('-c', '--command', help='following by a command or a program', nargs='+', default=[])
parser.add_argument('-v', '--verbose', help='verbose output', action='store_true', default=False)
parser.add_argument('-j', '--jobid', help='slurm job id', type=int)
parser.add_argument('-b', '--benchmark', help='name of benchmark', type=str)
parser.add_argument('-f', '--filename', help='events list in file',default='/usr/local/home/yyq/events_hard_pmu.txt')
parser.add_argument('-pa','--path',help='the path of shell script',default='/usr/local/home/yyq/mytest.sh') #measure specified shell
opts = parser.parse_args()
# print(opts)

# If verbose, it will output to both stdout and file
verbose = opts.verbose
# The tool will read counters per intvl ms.
intvl = opts.interval * 10
# Number of cores per socket
cps = 4  # core_per_socket
# print("Number of CPU cores per socket: %d" % cps)

# This block use Linux command 'perf' to monitor PMU counters

# filename = "events_list.txt"
with open(opts.filename) as events_file:
    events = events_file.read().splitlines()
    
eventsWithoutSpace = []
for e in events:
    eventsWithoutSpace.append(e.strip())

eventsWithoutSpace = sorted(set(eventsWithoutSpace),key=eventsWithoutSpace.index) # delete repeated events in list

events_str = ','.join(eventsWithoutSpace)
# Construct perf command
if opts.pid is not None:  # Monitor during a process (specified by PID)
    perf_cmd = 'perf stat -I %d -a -x, -e %s --pid %d' % (intvl, events_str, opts.pid)
else:  # Monitor during a command (if not empty)
    # perf_cmd = 'perf stat -I %d -a -x, -e %s %s -r 10 ./a.out' % (intvl, events_str, ' '.join(opts.command))
    perf_cmd = 'perf stat -I %d -a -x, -e %s ' % (intvl, events_str)


# The output file name
if opts.jobid is not None:
    fname = '/home/collect/data/micro/%s/pfmon-%d-%s.dat' % (opts.benchmark, opts.jobid, socket.gethostname())  # slurm job id, hostname    
else:
    fname = '/home/collect/data/micro/test/pfmon-%s.dat' % (socket.gethostname())  # hostname
fw = open(fname, 'a')  # append, not overwrite
if not fw:
    print("Cannot open file %s, Exit." % fname)
    exit(-1)

# Log perf command
if verbose:
    print(pfmon_color.blue + perf_cmd + pfmon_color.reset)
fw.write(perf_cmd);
fw.write('\n')


# Execute perf command
p = subprocess.Popen(perf_cmd, shell=True,stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=1)
#p = subprocess.Popen(perf_cmd, shell=True)
#p.send_signal(signal.SIGINT)
buf = []
# Print header
info_str = "Start at %s (UTC) on %s, Interval = %d ms" % (datetime.utcnow(), socket.gethostname(), intvl)
header_str = "YYYY-MM-DD HH:MM:SS.micros Insns(G)"
header_str=header_str+events_str
if verbose:
    print(pfmon_color.red + info_str + pfmon_color.reset)
fw.write(info_str);
fw.write('\n')
if verbose:
    print(pfmon_color.green + header_str + pfmon_color.reset)
    
#fw.write(header_str);
fw.write('\n')

# Parse perf output
for line in p.stderr:
	fw.write(line)

print('finish')

