#!/bin/bash
kill -9 $(pidof 'perf')
kill `ps -aux | grep 'dstat' | awk '{print $2}'`
rm -rf /var/lib/cni/networks/cbr0/*
