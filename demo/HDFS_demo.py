#!/usr/bin/env python
import sys
sys.path.append('../')
from logparser import Spell
import pandas as pd
from datetime import datetime

#parse HDFS log
input_dir  = '../../data/hdfs_loghub/HDFS_1/'  # The input directory of log file
output_dir = 'HDFS_result/'  # The output directory of parsing results
log_file   = 'HDFS.log'  # The input log file name
log_format = '<Date> <Time> <Pid> <Level> <Component>: <Content>'  # HDFS log format
tau        = 0.55  # Message type threshold (default: 0.5)
regex      = [
    "(/[-\w]+)+", #replace file path with *
    "(?<=blk_)[-\d]+" #replace block_id with *

]  # Regular expression list for optional preprocessing (default: [])

parser = Spell.LogParser(indir=input_dir, outdir=output_dir, log_format=log_format, tau=tau, rex=regex)
parser.parse(log_file)



# get [log key, delta time] as input for deeplog
# log_structured_file = output_dir + log_file + "_structured.csv"
# log_templates_file = output_dir + log_file + "_templates.csv"
#
# log_structured_df = pd.read_csv(log_structured_file)
# hdfs_input = []
# prev_time = None
# fmt = "%y%m%d %H%M%S"
# for idx, row in log_structured_df.iterrows():
#     if idx == 0:
#         delta_time = 0
#         prev_time = datetime.strptime("{0} {1}".format(row["Date"],row["Time"]), fmt)
#     else:
#         curr_time = datetime.strptime("{0} {1}".format(row["Date"],row["Time"]), fmt)
#         delta_time = (curr_time - prev_time).total_seconds()
#         prev_time = curr_time
#
#     log_key = row["EventId"]
#     hdfs_input.append([log_key, delta_time])
#
# print(pd.DataFrame(hdfs_input))