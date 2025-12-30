#!/bin/sh
# this is just an example of how to run the files locally.
# on a cluster, the enqueue command can be run on a login node,
# and the workers should be run on compute nodes.

dqmc-util enqueue queue data/*/*.h5

dqmc-util worker queue ../../build/dqmc -n 8 # 8 cpus. adjust as appropriate
# typical run time around 1 minute on laptop (i7-12700h) using 8 cpus
