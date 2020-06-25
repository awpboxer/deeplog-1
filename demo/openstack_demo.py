#!/usr/bin/env python
import sys
sys.path.append('../')
from logparser import Spell

input_dir  = '../../data/OpenStack/'  # The input directory of log file
output_dir = 'openstack_result/'  # The output directory of parsing results
# log_file   = 'openstack_abnormal.log'  # The input log file name
log_file   = 'openstack_normal1.log'  # The input log file name
# log_file   = 'openstack_normal2.log'  # The input log file name
log_format = '<Filename> <Date> <Time> <Pid> <Level> <Component> <Request_id> <Content>'  # openstack log format
tau        = 0.55  # Message type threshold (default: 0.5)
regex      = [
    #r"(?<=instance: )[A-Za-z0-9-]+" , replace id after instance with *
    r"([A-Za-z0-9]+-){4}[A-Za-z0-9]+", #replace instance id with *
    r"(/\w+)+" #replace file address /var/.../.. with *
]  # Regular expression list for optional preprocessing (default: []) for line in log


parser = Spell.LogParser(indir=input_dir, outdir=output_dir, log_format=log_format, tau=tau, rex=regex)
parser.has_list = True  # connect request ids in content with %
parser.parse(log_file)