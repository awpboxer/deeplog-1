import json
from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm


def read_json(filename):
    with open(filename, 'r') as load_f:
        file_dict = json.load(load_f)
    return file_dict


def trp(l, n):
    """ Truncate or pad a list """
    r = l[:n]
    if len(r) < n:
        r.extend(list([0]) * (n - len(r)))
    return r


def down_sample(logs, labels, sample_ratio):
    print('sampling...')
    total_num = len(labels)
    all_index = list(range(total_num))
    sample_logs = {}
    for key in logs.keys():
        sample_logs[key] = []
    sample_labels = []
    sample_num = int(total_num * sample_ratio)

    for i in tqdm(range(sample_num)):
        random_index = int(np.random.uniform(0, len(all_index)))
        for key in logs.keys():
            sample_logs[key].append(logs[key][random_index])
        sample_labels.append(labels[random_index])
        del all_index[random_index]
    return sample_logs, sample_labels


def sliding_window(data_dir, datatype, window_size, num_classes, sample_ratio=1):
    '''
    dataset structure
        result_logs(dict):
            result_logs['feature0'] = list()
            result_logs['feature1'] = list()
            ...
        labels(list)
    '''
    #event2semantic_vec = read_json(data_dir + 'hdfs/event2semantic_vec.json')
    num_sessions = 0
    result_logs = {}
    result_logs['Sequentials'] = []
    result_logs['Quantitatives'] = []
    result_logs['Semantics'] = []
    result_logs['Parameters'] = []
    labels = []
    if datatype == 'val':
        data_dir += 'test_normal'
    else:
        data_dir += f'{datatype}'

    sample_size = 1000
    if sample_ratio != 1:
        print("Sampling")
        f = open(data_dir, 'r')
        sample_size = int(len(f.readlines()) * sample_ratio)
        f.close()
        print("sample size", sample_size)

    #7 8 10 9\n  log keys
    #[7,10] [8,2] [10,5] [9,6]\n log keys and params(eg: time)
    with open(data_dir, 'r') as f:
        for line in f.readlines():
            if len(line.strip()) == 0:
                continue

            num_sessions += 1
            if sample_ratio != 1 and num_sessions > sample_size:
                break

            if num_sessions % 1000 == 0:
                print("processed %s lines"%num_sessions, end='\r')

            # split log keys and other params(features)
            line = list(map(eval, line.strip().strip('"').split()))
            params = []
            if isinstance(line[0], int):
                line = tuple(map(lambda n: n - 1, line))
            else:
                params = tuple(map(lambda x: x[1], line))
                line = tuple(map(lambda x: x[0]-1, line))

            #line = tuple(map(lambda n: n - 1, map(int, line.strip().split())))

            for i in range(len(line) - window_size):
                Sequential_pattern = list(line[i:i + window_size])
                Quantitative_pattern = [0] * num_classes
                Parameter_pattern = list(params[i:i+window_size]) if params else [0]*window_size
                log_counter = Counter(Sequential_pattern)

                for key in log_counter:
                    Quantitative_pattern[key] = log_counter[key]
                Semantic_pattern = []
                # for event in Sequential_pattern:
                #     if event == 0:
                #         Semantic_pattern.append([-1] * 300)
                #     else:
                #         Semantic_pattern.append(event2semantic_vec[str(event -
                #                                                        1)])
                Sequential_pattern = np.array(Sequential_pattern)[:,
                                                               np.newaxis]
                Quantitative_pattern = np.array(
                    Quantitative_pattern)[:, np.newaxis]

                Parameter_pattern = np.array(Parameter_pattern)[:, np.newaxis] #increase one more dimension
                result_logs['Sequentials'].append(Sequential_pattern)
                result_logs['Quantitatives'].append(Quantitative_pattern)
                result_logs['Semantics'].append(Semantic_pattern)
                result_logs["Parameters"].append(Parameter_pattern)
                labels.append(line[i + window_size])
    #
    # if sample_ratio != 1:
    #     result_logs, labels = down_sample(result_logs, labels, sample_ratio)

    print('File {}, number of sessions {}'.format(data_dir, num_sessions))
    print('File {}, number of seqs {}'.format(data_dir,
                                              len(result_logs['Sequentials'])))

    return result_logs, labels


def session_window(data_dir, datatype, sample_ratio=1):
    event2semantic_vec = read_json(data_dir + 'hdfs/event2semantic_vec.json')
    result_logs = {}
    result_logs['Sequentials'] = []
    result_logs['Quantitatives'] = []
    result_logs['Semantics'] = []
    labels = []

    if datatype == 'train':
        data_dir += 'hdfs/robust_log_train.csv'
    elif datatype == 'val':
        data_dir += 'hdfs/robust_log_valid.csv'
    elif datatype == 'test':
        data_dir += 'hdfs/robust_log_test.csv'

    train_df = pd.read_csv(data_dir)
    for i in tqdm(range(len(train_df))):
        ori_seq = [
            int(eventid) for eventid in train_df["Sequence"][i].split(' ')
        ]
        Sequential_pattern = trp(ori_seq, 50)
        Semantic_pattern = []
        for event in Sequential_pattern:
            if event == 0:
                Semantic_pattern.append([-1] * 300)
            else:
                Semantic_pattern.append(event2semantic_vec[str(event - 1)])
        Quantitative_pattern = [0] * 29
        log_counter = Counter(Sequential_pattern)

        for key in log_counter:
            Quantitative_pattern[key] = log_counter[key]

        Sequential_pattern = np.array(Sequential_pattern)[:, np.newaxis]
        Quantitative_pattern = np.array(Quantitative_pattern)[:, np.newaxis]
        result_logs['Sequentials'].append(Sequential_pattern)
        result_logs['Quantitatives'].append(Quantitative_pattern)
        result_logs['Semantics'].append(Semantic_pattern)
        labels.append(int(train_df["label"][i]))

    if sample_ratio != 1:
        result_logs, labels = down_sample(result_logs, labels, sample_ratio)

    # result_logs, labels = up_sample(result_logs, labels)

    print('Number of sessions({}): {}'.format(data_dir,
                                              len(result_logs['Semantics'])))
    return result_logs, labels
