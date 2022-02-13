#!/bin/bash
kill `ps -aux | grep 'tpcdsMulti.py' | awk '{print $2}'`
sleep 3
kill  `ps -aux | grep 'org.apache.spark.deploy.SparkSubmit' | awk '{print $2}'`