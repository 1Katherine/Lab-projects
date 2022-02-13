#!/usr/bin/python3
# NOTE!!! This monitor script only works for Intel Xeon E5/E7 series
# For scalable platforms (starts from skylake), no support yet.
# To use this script, one needs to modify the 'cps' variable according to the CPU
import subprocess
import argparse

# Parse input options
parser = argparse.ArgumentParser(description="A PMU Monitor.")
parser.add_argument('-i', '--interval', help='interval to read counters', type=int, default=10)
parser.add_argument('-p', '--pid', help='attach to a process by pid', type=int)
parser.add_argument('-c', '--command', help='following by a command or a program', nargs='+', default=[])
parser.add_argument('-j', '--jobid', help='slurm job id', type=int)
parser.add_argument('-pa', '--path', help='the path of shell script', type=str,
                    default='/usr/local/home/yyq/mytest.sh')  # measure specified shell
opts = parser.parse_args()

# The tool will read counters per intvl ms.
intvl = opts.interval * 100


# Construct perf command
perf_cmd = 'perf stat -I %d -a -x, -e branch-instructions bash %s' % (intvl, opts.path)

print("cmd\n")

# Execute perf command
p = subprocess.Popen(perf_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=1)

# Parse perf output
print("end before line")
for line in p.stderr:
    pass

print("finish")
