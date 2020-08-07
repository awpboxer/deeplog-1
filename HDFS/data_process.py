import sys
sys.path.append('../')

import os
import gc
import platform
import pandas as pd
import argparse
from tqdm import tqdm
import re
import numpy as np
from collections import OrderedDict

data_dir = "../../data/hdfs_loghub/"
output_dir = "../output/hdfs/"

def session_window(log_df, cols):
    print("get sequence by session window")
    data_dict = OrderedDict() #preserve insertion order of items
    for idx, row in tqdm(log_df.iterrows()):
        blkId_list = re.findall(r'(blk_-?\d+)', row['Content'])
        blkId_set = set(blkId_list)
        for blk_Id in blkId_set:
            if not blk_Id in data_dict:
                data_dict[blk_Id] = [row[cols]]
            else:
                data_dict[blk_Id].append(row[cols])

    seq = []
    for k, v in data_dict.items():
        temp = [k]
        for col in cols:
            temp.append([t[col] for t in v])
        seq.append(temp)
    return pd.DataFrame(seq, columns=["BlockId"] + cols)

def deeplog_file_generator(filename, df, features):
    with open(filename, 'w') as f:
        for _, row in df.iterrows():
            for val in zip(*row[features]):
                f.write(','.join([str(v) for v in val]) + ' ')
            f.write('\n')

def generate_train_test(input_dir, seq, cols, n=None, train_ratio=0.3):
    print("===========Generate train test data==============")
    blk_label_dict = {}
    blk_label_file = os.path.join(input_dir, "anomaly_label.csv") ####???
    blk_df = pd.read_csv(blk_label_file)
    for _, row in tqdm(blk_df.iterrows()):
        blk_label_dict[row["BlockId"]] = 1 if row["Label"] == "Anomaly" else 0

    seq["Label"] = seq["BlockId"].apply(lambda x: blk_label_dict.get(x))

    #########
    # Train #
    #########
    normal_seq = seq[seq["Label"] == 0][cols]
    normal_seq = normal_seq.sample(frac=1, random_state=20) # shuffle normal data
    normal_len = len(normal_seq)
    train_len = n if n else int(normal_len * train_ratio)
    deeplog_file_generator(os.path.join(output_dir,'train'), normal_seq.iloc[:train_len],  cols)
    print("training size {}".format(train_len))

    ###############
    # Test Normal #
    ###############
    deeplog_file_generator(os.path.join(output_dir, 'test_normal'),normal_seq.iloc[train_len:], cols)
    print("test normal size {}".format(normal_len - train_len))


    #################
    # Test Abnormal #
    #################
    abnormal_seq = seq[seq["Label"] == 1][cols]
    deeplog_file_generator(os.path.join(output_dir,'test_abnormal'), abnormal_seq, cols)
    print('test abnormal size {}'.format(len(abnormal_seq)))

    print("==================Done================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', choices=[1,2], type=int, help="log file: 1=sample data, 2=data")
    parser.add_argument('-l', type=bool, help='if start log parser')
    parser.add_argument('-p', default='drain', type=str, help="parser type")
    parser.add_argument('-r', default=0.4, type=float, help="train ratio")
    parser.add_argument('-col', default='1', type=str, help='column: 1=log key, 2=timestamp')
    if platform.system() == 'Windows':
        args = parser.parse_args('-f 2 -col 12'.split())
    else:
        args = parser.parse_args()
    print(args)

    if args.f == 1:
        log_file = "HDFS_2K.log.txt"
    elif args.f == 2:
        data_dir += "HDFS_1/"
        log_file = "HDFS.log"
    else:
        raise AttributeError('log file is not defined')


    ##################
    # Transformation #
    ##################
    df = pd.read_csv(f'{output_dir}{log_file}_structured.csv', dtype={'Date':object, "Time": object})
    cols = []
    if "1" in args.col:
        cols.append('EventId')
        df_temp = pd.read_csv(f'{output_dir}{log_file}_templates.csv')
        event_id_map = dict()
        for i, event_id in enumerate(df_temp['EventId'].unique(), 1):
            event_id_map[event_id] = i
        print(f'length of event_id_map: {len(event_id_map)}')

        df.loc[:,"EventId"] = df["EventId"].apply(lambda e: event_id_map[e] if event_id_map.get(e) else -1)

    if "2" in args.col:
        df["Datetime"] = df["Date"] + " " + df["Time"]
        df["Datetime"] = pd.to_datetime(df["Datetime"], format='%y%m%d %H%M%S')
        cols.append('deltaT')
        df['deltaT'] = df['Datetime'].diff().dt.seconds
        df['deltaT'] = df['deltaT'].apply(lambda t: t if t > 0 else abs(np.random.normal(0, 1)))

    # df["blk"] = df["Content"].apply(lambda x: re.findall(r'(blk_-?\d+)', x))
    # df["blk"].dropna()
    # log_df = df[['blk'] + cols].groupby("blk").apply(list).reset_index()

    if not cols:
        raise NotImplementedError

    seq = session_window(df, cols)
    generate_train_test(data_dir, seq, cols, n=4855, train_ratio=args.r)