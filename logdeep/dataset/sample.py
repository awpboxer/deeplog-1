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

# given window size, generate sequences. one session(one line) can generate session_size-window_size sequences
def sliding_window(data_dir, log_name, datatype, window_size, num_of_class, sample_ratio=1):
    '''
    dataset structure
        result_logs(dict):
            result_logs['feature0'] = list()
            result_logs['feature1'] = list()
            ...
        labels(list)
    '''
    # event2semantic_vec = read_json(data_dir + 'hdfs/event2semantic_vec.json')  #mapping event(class) to a vector like word embedding
    num_sessions = 0 #number of lines in log data
    result_logs = {}
    result_logs['Sequentials'] = [] #sequence data eg: [k4,k4,k5,k6,k8]
    result_logs['Quantitatives'] = [] #one hot encode sequence data eg: variables = [k1,k2,k3,....k8][0,0,0,2,1,1,0,1]
    result_logs['Semantics'] = [] #mapping value to a vector like word embedding.
    labels = []
    if datatype == 'train':
        data_dir +=f"{log_name}/{log_name}_{datatype}.csv" # eg: 'hdfs/hdfs_train'
    if datatype == 'val':
        data_dir += f"{log_name}/{log_name}_test_normal.csv"
    # data_dir += f"{log_name}/{log_name}_{datatype}.csv"

    with open(data_dir, 'r') as f:
        for line in f.readlines():
            # use small size of data
            if num_sessions > 20:
                break

            num_sessions += 1
            line = tuple(map(lambda n: n - 1, map(int, line.strip().split())))

            for i in range(len(line) - window_size):
                Sequential_pattern = list(line[i:i + window_size])
                Quantitative_pattern = [0] * num_of_class  #dim_size stands for number of classes
                log_counter = Counter(Sequential_pattern)

                for key in log_counter:
                    Quantitative_pattern[key] = log_counter[key]
                Semantic_pattern = []
                # for event in Sequential_pattern:
                #     if event == 0:
                #         Semantic_pattern.append([-1] * 300) #using a 300 dimension vector to represent the value
                #     else:
                #         Semantic_pattern.append(event2semantic_vec[str(event -
                #                                                        1)])
                # Sequential_pattern = np.array(Sequential_pattern)[:,
                #                                                   np.newaxis] #np newaxis add one more dimension, here add column vector

                Quantitative_pattern = np.array(
                    Quantitative_pattern)[:, np.newaxis]
                result_logs['Sequentials'].append(Sequential_pattern)
                result_logs['Quantitatives'].append(Quantitative_pattern)
                result_logs['Semantics'].append(Semantic_pattern)
                labels.append(line[i + window_size]) #next_value

    if sample_ratio != 1:
        result_logs, labels = down_sample(result_logs, labels, sample_ratio)

    print('File {}, number of sessions {}'.format(data_dir, num_sessions))
    print('File {}, number of seqs {}'.format(data_dir,
                                              len(result_logs['Sequentials'])))

    return result_logs, labels

# in hdfs, one line is a session. a sequence with fixed size 50
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
        Sequential_pattern = trp(ori_seq, 50)  #fixed length of 50
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
