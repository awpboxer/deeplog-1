import pandas as pd
import os

input_dir  = '../data/hdfs_loghub/HDFS_1/'  # The input directory of log file
log_file   = 'HDFS.log'

file_path = os.path.join(input_dir,log_file)
print(file_path)

time_info = []
with open(file_path, "r" ) as f:
    for line in f.readlines():
        line = line.split(" ")
        time_info.append(line[:2])
        print(line)
        break

time_df = pd.DataFrame(time_info, columns = ["date", "time"])

print(time_df.head())

