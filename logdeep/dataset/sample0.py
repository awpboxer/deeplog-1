import json
from collections import Counter
from copy import deepcopy
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


    data_dir += f"{log_name}_result/{log_name}_log_label.csv"
    with open(data_dir, 'r') as f:
        for line in f.readlines():
            # use small size of data
            if num_sessions > 20:
                break

            num_sessions += 1
            #line = tuple(map(lambda n: n - 1, map(int, line.strip().split(","))))

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
def session_window(data_dir, log_name,window_size , sample_ratio=1):
    #event2semantic_vec = read_json(data_dir + 'hdfs/event2semantic_vec.json')
    result_logs = {}
    result_logs['Sequentials'] = []
    result_logs['Quantitatives'] = []
    result_logs['Semantics'] = []

    normal_result_logs = deepcopy(result_logs)
    abnormal_result_logs = deepcopy(result_logs)

    normal_labels = []
    abnormal_labels = []

    with open(data_dir+"hdfs_log_templates.json", "r") as f:
        log_temp = json.load(f)

    data_dir += f"{log_name}_log_label1.csv"

    train_df = pd.read_csv(data_dir)
    #train_df.drop(train_df.columns[0], axis=1, inplace=True)
    #train_df.to_csv(data_dir, index=False)
    label = train_df["label"]

    # mapping event id to number
    print("mapping list")
    event = train_df["EventId"].apply(lambda x: log_temp[x])

    #mapping event id to one hot
    #event_list = list(log_temp.keys())
    #event = train_df["EventId"].apply(lambda x: [1 if x == e else 0 for e in event_list])

    features = list(event)

    #train_df.drop("label", axis=1, inplace=True)
    for i in tqdm(range(0,len(train_df)-window_size, window_size)):
        label_seq = label[i:i+window_size] #a sequence is abnormal if anonmaly exists in a sequence
        Sequential_pattern = features[i: i+window_size]
        if len(Sequential_pattern) < window_size or i + window_size >= len(train_df):
            #Sequential_pattern = Sequential_pattern + [0]*(window_size - len(Sequential_pattern)) #padding
            break
        Semantic_pattern = []
        #for event in Sequential_pattern:
        #    if event == 0:
        #        Semantic_pattern.append([-1] * 300)
        #    else:
        #        Semantic_pattern.append(event2semantic_vec[str(event - 1)])
        Quantitative_pattern = [0] * 29
        #log_counter = Counter(Sequential_pattern)

        #for key in log_counter:
        #    Quantitative_pattern[key] = log_counter[key]

        Sequential_pattern = np.array(Sequential_pattern)[:, np.newaxis]
        #Quantitative_pattern = np.array(Quantitative_pattern)[:, np.newaxis]

        is_anomaly = max(label_seq)
        if is_anomaly == 0: # sequence is anomaly =1, no = 0
            normal_result_logs['Sequentials'].append(Sequential_pattern)
            normal_result_logs['Quantitatives'].append(Quantitative_pattern)
            normal_result_logs['Semantics'].append(Semantic_pattern)
            normal_labels.append(features[i+window_size])
        else:
            abnormal_result_logs['Sequentials'].append(Sequential_pattern)
            abnormal_result_logs['Quantitatives'].append(Quantitative_pattern)
            abnormal_result_logs['Semantics'].append(Semantic_pattern)
            abnormal_labels.append(features[i+window_size])

    print("normal log size is {0}, abnormal log size is {1}".format(len(normal_result_logs["Sequentials"]), len(abnormal_result_logs["Sequentials"])))

    #result_logs, labels = down_sample(result_logs, labels, sample_ratio)
    train_logs = result_logs.copy()
    test_normal_logs = result_logs.copy()
    test_abnormal_logs = result_logs.copy()


    normal_size = len(normal_result_logs["Sequentials"])
    abnormal_size = len(abnormal_result_logs["Sequentials"])*1

    test_normal_size = min(int(normal_size*0.5), abnormal_size)
    sample_size = (normal_size - test_normal_size) * sample_ratio

    train_logs["Sequentials"] = normal_result_logs["Sequentials"][:sample_size]
    train_logs["Quantitatives"] = normal_result_logs["Quantitatives"][:sample_size]
    train_logs["Semantics"] = normal_result_logs["Semantics"][:sample_size]
    train_labels = normal_labels[:sample_size]

    test_normal_logs["Sequentials"] = normal_result_logs["Sequentials"][sample_size:sample_size+test_normal_size]
    test_normal_logs["Quantitatives"] = normal_result_logs["Quantitatives"][sample_size:sample_size+test_normal_size]
    test_normal_logs["Semantics"] = normal_result_logs["Semantics"][sample_size:sample_size+test_normal_size]
    test_normal_labels = normal_labels[sample_size:sample_size+test_normal_size]

    test_abnormal_logs["Sequentials"] = abnormal_result_logs["Sequentials"][:abnormal_size]
    test_abnormal_logs["Quantitatives"] = abnormal_result_logs["Quantitatives"][:abnormal_size]
    test_abnormal_logs["Semantics"] = abnormal_result_logs["Semantics"][:abnormal_size]
    test_abnormal_labels = abnormal_labels[:abnormal_size]

    print("train size is {0}, test normal size is {1}, test abnormal size is {2}".format(len(train_labels), len(test_normal_labels), len(test_abnormal_labels)))
    return train_logs, train_labels, test_normal_logs, test_normal_labels, test_abnormal_logs, test_abnormal_labels

