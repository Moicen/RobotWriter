import collections
import os
import sys
import numpy as np

log_dir = "./log"
log_file = "./log/log.txt"

def log(label, info):
	if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        with open(log_file, 'w') as f:
            f.write("%s : %s" % label, str(info))