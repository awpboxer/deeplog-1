import os
import sys
import gc
import logging
import pandas as pd
from OpenStack import const
from logparser import Spell, Drain

pd.options.mode.chained_assignment = None  # default='warn'

logging.basicConfig(level=logging.WARNING,
                    format='[%(asctime)s][%(levelname)s]: %(message)s')
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))

data_dir =  "../../data/BGL/"
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


def deeplog_df_transfer(df, event_id_map, window_size='T'):
    df['datetime'] = pd.to_datetime(df['Time'], format='%Y-%m-%d-%H.%M.%S.%f')
    df = df[['datetime', 'EventId','Label']]
    df.loc[:,"EventId"] = df['EventId'].apply(lambda e: event_id_map[e] if event_id_map.get(e) else -1)
    df["Label"] = df["Label"].apply(lambda x: int(x != "-"))
    deeplog_df = df.set_index('datetime').resample(window_size).agg({"label": "max", "EventId": _custom_resampler}).reset_index()
    return deeplog_df


def _custom_resampler(array_like):
    return list(array_like)


def deeplog_file_generator(filename, df):
    with open(filename, 'w') as f:
        for _, row in df.iterrows():
            event_id_list = row['EventId']
            for event_id in event_id_list:
                f.write(str(event_id) + ' ')
            f.write(',' + str(row["Label"]))
            f.write('\n')

def parse_log(input_dir, output_dir, log_file, parser_type):
    log_format = '<Label> <Id> <Date> <Code1> <Time> <Code2> <Component1> <Component2> <Level> <Content>'
    regex = []
    keep_para = False
    if parser_type == "drain":
        # the hyper parameter is set according to http://jmzhu.logpai.com/pub/pjhe_icws2017.pdf
        st = 0.5  # Similarity threshold
        depth = 4  # Depth of all leaf nodes
        parser = Drain.LogParser(log_format, indir=input_dir, outdir=output_dir, depth=depth, st=st, rex=regex, keep_para=keep_para)
        parser.parse(log_file)
    elif parser_type == "spell":
        tau = 0.55
        parser = Spell.LogParser(indir=data_dir, outdir=output_dir, log_format=log_format, tau=tau, rex=regex, keep_para=keep_para)
        parser.parse(log_file)


if __name__ == "__main__":
    #count_anomaly()

    ##########
    # Parser #
    #########
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # parse_log(data_dir, output_dir, log_file, 'drain')
    # sys.exit()

    ##################
    # Transformation #
    ##################
    df = pd.read_csv(f'{output_dir}{log_file}_structured.csv')

    df_temp = pd.read_csv(f'{output_dir}{log_file}_templates.csv')
    event_id_map = dict()
    for i, event_id in enumerate(df_temp['EventId'].unique(), 1):
        event_id_map[event_id] = i
    logger.info(f'length of event_id_map: {len(event_id_map)}')

    deeplog_df = deeplog_df_transfer(df, event_id_map, window_size='T')

    #########
    # Train #
    #########
    df_normal =deeplog_df[deeplog_df["label"] == 0]
    df_normal = df_normal.sample(frac=1).reset_index(drop=True) #shuffle
    normal_len = len(df_normal)
    train_len = int(normal_len * 0.01)
    deeplog_file_generator(os.path.join(output_dir,'train'), df_normal[:train_len])
    logger.info("training size {}".format(train_len))

    ###############
    # Test Normal #
    ###############
    deeplog_file_generator(os.path.join(output_dir, 'test_normal'), df_normal[train_len:])
    logger.info("test normal size {}".format(normal_len - train_len))

    del df_normal
    gc.collect()

    #################
    # Test Abnormal #
    #################
    df_abnormal = deeplog_df[deeplog_df["label"] == 1]
    deeplog_file_generator(os.path.join(output_dir,'test_abnormal'), df_abnormal)
    logger.info('test abnormal size {}'.format(len(df_abnormal)))