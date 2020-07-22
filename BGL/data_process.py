import sys
sys.path.append('../')

import os
import gc
import logging
import pandas as pd
from logparser import Spell, Drain
import argparse
from tqdm import tqdm

tqdm.pandas()

pd.options.mode.chained_assignment = None  # default='warn'

logging.basicConfig(level=logging.WARNING,
                    format='[%(asctime)s][%(levelname)s]: %(message)s')
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))

data_dir = "../../data/BGL/"
log_file = "BGL_2k.log.txt"
output_dir = "../output/bgl/"

#In the first column of the log, "-" indicates non-alert messages while others are alert messages.
def count_anomaly():
    total_size = 0
    normal_size = 0
    with open(data_dir + log_file, encoding="utf8") as f:
        for line in f.readlines():
            total_size += 1
            if line.split(' ',1)[0] == '-':
                normal_size += 1
    print("total size {}, abnormal size {}".format(total_size, total_size - normal_size))


def deeplog_df_transfer(df, features, window_size='T'):
    """
    :param window_size: offset datetime https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects
    :return:
    """
    cols = ['datetime','Label'] + features
    agg_dict = {'Label':'max'}
    for f in features:
        agg_dict[f] = _custom_resampler

    df = df[cols]
    df["Label"] = df["Label"].apply(lambda x: int(x != "-"))
    deeplog_df = df.set_index('datetime').resample(window_size).agg(agg_dict).reset_index()
    return deeplog_df


def _custom_resampler(array_like):
    return list(array_like)


def deeplog_file_generator(filename, df, features):
    with open(filename, 'w') as f:
        for _, row in df.iterrows():
            for val in zip(*row[features]):
                f.write(','.join([str(v) for v in val]) + ' ')
            f.write('\n')


def parse_log(input_dir, output_dir, log_file, parser_type):
    log_format = '<Label> <Id> <Date> <Code1> <Time> <Code2> <Component1> <Component2> <Level> <Content>'
    regex = [
        r'(0x)[0-9a-fA-F]+', #hexadecimal
        r'[0-9a-fA-F]{8}', #number and its next characters, e.g.00544eb8
        r'\d+'
    ]
    keep_para = False
    if parser_type == "drain":
        # the hyper parameter is set according to http://jmzhu.logpai.com/pub/pjhe_icws2017.pdf
        st = 0.3  # Similarity threshold
        depth = 3  # Depth of all leaf nodes
        parser = Drain.LogParser(log_format, indir=input_dir, outdir=output_dir, depth=depth, st=st, rex=regex, keep_para=keep_para)
        parser.parse(log_file)
    elif parser_type == "spell":
        tau = 0.55
        parser = Spell.LogParser(indir=data_dir, outdir=output_dir, log_format=log_format, tau=tau, rex=regex, keep_para=keep_para)
        parser.parse(log_file)


def compute_time_delta(time_list):
    print("time list length", len(time_list))
    time_list_ = list(map(lambda x: (x - time_list[0]).seconds, time_list))
    #t_max = max(time_list_)
    #time_list_ = [round((t)/(t_max + 0.0001), 4) for t in time_list_]
    return [t for t in time_list_ if t > 0 ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', choices=[1,2], type=int, help="log file: 1=sample data, 2=data")
    parser.add_argument('-l', type=bool, help='if start log parser')
    parser.add_argument('-p', default='drain', type=str, help="parser type")
    parser.add_argument('-w', default='T', type=str, help='window size')
    parser.add_argument('-r', default=0.4, type=float, help="train ratio")
    parser.add_argument('-col', default='1', type=str, help='column: 1=log key, 2=timestamp')
    args = parser.parse_args('-f 2 -w 30S -col 2'.split())
    print(args)
    if args.f == 1:
        log_file = "BGL_2k.log.txt"
    elif args.f == 2:
        log_file = "BGL.log"
    else:
        raise AttributeError('log file is not defined')

    ##########
    # Parser #
    #########
    if args.l:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        parse_log(data_dir, output_dir, log_file, args.p)
        sys.exit()

    #########
    # Count #
    #########
    # count_anomaly()

    ##################
    # Transformation #
    ##################
    df = pd.read_csv(f'{output_dir}{log_file}_structured.csv')
    df['datetime'] = pd.to_datetime(df['Time'], format='%Y-%m-%d-%H.%M.%S.%f')


    cols = []
    if "1" in args.col:
        cols.append('EventId')
        df_temp = pd.read_csv(f'{output_dir}{log_file}_templates.csv')
        event_id_map = dict()
        for i, event_id in enumerate(df_temp['EventId'].unique(), 1):
            event_id_map[event_id] = i
        print(f'length of event_id_map: {len(event_id_map)}')

        df.loc[:,"EventId"] = df["EventId"].progress_apply(lambda e: event_id_map[e] if event_id_map.get(e) else -1)

    if "2" in args.col:
        cols.append('deltaT')
        df['deltaT'] = df['datetime'].diff().dt.microseconds.div(1000000, fill_value=0) #microseconds / 1000000 = seconds

    if not cols:
        raise NotImplementedError

    deeplog_df = deeplog_df_transfer(df, cols, window_size=args.w)
    deeplog_df.dropna(subset=["Label"], inplace=True)

    # if "2" in args.col:
    #     deeplog_df["datetime_"] = deeplog_df["datetime_"].apply(compute_time_delta)

    #########
    # Train #
    #########
    df_normal =deeplog_df[deeplog_df["Label"] == 0]
    df_normal = df_normal.sample(frac=1).reset_index(drop=True) #shuffle
    normal_len = len(df_normal)
    train_ratio = args.r
    train_len = int(normal_len * train_ratio)
    deeplog_file_generator(os.path.join(output_dir,'train'), df_normal[:train_len], cols)
    print("training size {}".format(train_len))

    ###############
    # Test Normal #
    ###############
    deeplog_file_generator(os.path.join(output_dir, 'test_normal'), df_normal[train_len:], cols)
    print("test normal size {}".format(normal_len - train_len))

    del df_normal
    gc.collect()

    #################
    # Test Abnormal #
    #################
    df_abnormal = deeplog_df[deeplog_df["Label"] == 1]
    deeplog_file_generator(os.path.join(output_dir,'test_abnormal'), df_abnormal, cols)
    print('test abnormal size {}'.format(len(df_abnormal)))