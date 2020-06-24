import os
import re
import numpy as np
import pandas as pd
from collections import OrderedDict
from HDFS import hdfs_const

def hdfs_sampling(log_file, window='session', window_size=0):
    assert window == 'session', "Only window=session is supported for HDFS dataset."
    print("Loading", log_file)
    struct_log = pd.read_csv(log_file, engine='c',
            na_filter=False, memory_map=True)
    #print("log length", len(struct_log))
    #return
    data_dict = OrderedDict() #preserve insertion order of items
    for idx, row in struct_log.iterrows():
        blkId_list = re.findall(r'(blk_-?\d+)', row['Content'])
        blkId_set = set(blkId_list)
        for blk_Id in blkId_set:
            if not blk_Id in data_dict:
                data_dict[blk_Id] = []
            data_dict[blk_Id].append(row['EventId'])
    data_df = pd.DataFrame(list(data_dict.items()), columns=['BlockId', 'EventSequence'])
    data_df.to_csv(f"{output_dir}{hdfs_const.PARSER}_result/hdfs_sequence.csv",index=None)

output_dir = "../output/hdfs/"
hdfs_sampling(f"{output_dir}{hdfs_const.PARSER}_result/HDFS.log_structured.csv")