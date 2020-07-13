#https://github.com/nailo2c/deeplog/blob/f498ed6c94feb6e36ba25c835911c0653bd89e5e/example/preprocess.py#L25

import os
import sys
import logging
import pandas as pd
from OpenStack import const
from logparser import Spell

pd.options.mode.chained_assignment = None  # default='warn'

logging.basicConfig(level=logging.WARNING,
                    format='[%(asctime)s][%(levelname)s]: %(message)s')
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))


def deeplog_df_transfer(df, event_id_map):
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df = df[['datetime', 'EventId']]
    df.loc[:,"EventId"] = df['EventId'].apply(lambda e: event_id_map[e] if event_id_map.get(e) else -1)
    deeplog_df = df.set_index('datetime').resample('1min').apply(_custom_resampler).reset_index()
    return deeplog_df


def _custom_resampler(array_like):
    return list(array_like)


def deeplog_file_generator(filename, df):
    with open(filename, 'w') as f:
        for event_id_list in df['EventId']:
            for event_id in event_id_list:
                f.write(str(event_id) + ' ')
            f.write('\n')


if __name__ == '__main__':
    ##########
    # Parser #
    ##########
    input_dir = const.DATA_DIR
    output_dir = const.OUTPUT_DIR + const.PARSER + "_result2/"  # The output directory of parsing results

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    params = {
        'log_format': '<Logrecord> <Date> <Time> <Pid> <Level> <Component> \[<ADDR>\] <Content>',
        'tau': 0.9,
        'keep_para':False,
        'regex':[
            r"(?<=\[instance: )[A-Za-z0-9-]+",  # instance id
            r"\w+-\w+-\w+-\w+-\w+", #instance id xx-xxx-xx-xx-xx
            r'((\d+\.){3}\d+,?)+',  # ip address
            r'/.+?\s',  # /file path
            r'[\d.]+'  # numbers
        ]
        }
    # parser = Spell.LogParser(indir=input_dir, outdir=output_dir, log_format=params['log_format'], tau=params['tau'], rex=params['regex'], keep_para=params["keep_para"])
    #
    # for log_name in ['openstack_abnormal.log', 'openstack_normal2.log', 'openstack_normal1.log']:
    #     parser.parse(log_name)
    #
    # sys.exit()
    ##################
    # Transformation #
    ##################
    df = pd.read_csv(f'{output_dir}/openstack_normal1.log_structured.csv')
    df_normal = pd.read_csv(f'{output_dir}/openstack_normal2.log_structured.csv')
    df_abnormal = pd.read_csv(f'{output_dir}/openstack_abnormal.log_structured.csv')

    event_id_map = dict()
    for i, event_id in enumerate(df['EventId'].unique(), 1):
        event_id_map[event_id] = i

    logger.info(f'length of event_id_map: {len(event_id_map)}')

    #########
    # Train #
    #########
    deeplog_train = deeplog_df_transfer(df, event_id_map)
    deeplog_file_generator(os.path.join(output_dir,'train'), deeplog_train)

    ###############
    # Test Normal #
    ###############
    deeplog_test_normal = deeplog_df_transfer(df_normal, event_id_map)
    deeplog_file_generator(os.path.join(output_dir, 'test_normal'), deeplog_test_normal)

    #################
    # Test Abnormal #
    #################
    deeplog_test_abnormal = deeplog_df_transfer(df_abnormal, event_id_map)
    deeplog_file_generator(os.path.join(output_dir,'test_abnormal'), deeplog_test_abnormal)