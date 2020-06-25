
import os
import re
import json
import pandas as pd
from collections import OrderedDict
from OpenStack import const
from datetime import datetime


# get [log key, delta time] as input for deeplog
input_dir  = const.DATA_DIR
output_dir = const.OUTPUT_DIR
abnormal_log_file   = 'openstack_abnormal.log'
normal1_log_file = 'openstack_normal1.log'
normal2_log_file = 'openstack_normal2.log'

log_file = 'OpenStack_2k.log' #'openstack.log'
output_dir += const.PARSER + "_result/"

log_structured_file = output_dir + log_file + "_structured.csv"
log_templates_file = output_dir + log_file + "_templates.csv"
log_sequence_file = output_dir + "openstack_sequence.csv"


def concat_logs(files, input_dir, output_dir):
    data = []
    for f in files:
        with open(input_dir + f, 'r') as ff:
            data += ff.readlines()
    with open(output_dir,'w') as f:
        f.writelines(data)
    print("concat logs done")

# mapping eventid to number
def mapping():
    log_temp = pd.read_csv(log_templates_file)
    log_temp.sort_values(by = ["Occurrences"], ascending=False, inplace=True)
    log_temp_dict = {event: idx+1 for idx , event in enumerate(list(log_temp["EventId"])) }
    print(log_temp_dict)
    with open (output_dir + "log_templates.json", "w") as f:
        json.dump(log_temp_dict, f)


def sampling(log_file, window='session', window_size=0):
    assert window == 'session', "Only window=session is supported for HDFS dataset."
    print("Loading", log_file)
    struct_log = pd.read_csv(log_file, engine='c',
            na_filter=False, memory_map=True)

    with open(output_dir + "log_templates.json", "r") as f:
        event_num = json.load(f)

    data_dict = OrderedDict() #preserve insertion order of items
    for idx, row in struct_log.iterrows():
        sessionId_list = re.findall(r'(?<=\[instance: )[A-Za-z0-9-]+', row['Content'])
        sessionId_set = set(sessionId_list) #in openstack, session id is instance id
        for session_Id in sessionId_set:
            if not session_Id in data_dict:
                data_dict[session_Id] = []
            num = event_num.get(row["EventId"])
            if num:
                data_dict[session_Id].append([num, row['Date'], row['Time']])
            else:
                print("No mapping for EventId: %s"%row["EventId"])
                print(row)
    data_df = pd.DataFrame(list(data_dict.items()), columns=['EventId', 'EventSequence'])
    data_df.to_csv(log_sequence_file,index=None)
    print("sampling done")


def compute_time_diff(sequence):
    sequence = eval(sequence)
    new_sequence = ''
    fmt = "%Y-%m-%d %H:%M:%S.%f"
    prev_time = datetime.strptime("{0} {1}".format(sequence[0][1], sequence[0][2]), fmt)
    for s in sequence:
        curr_time = datetime.strptime("{0} {1}".format(s[1], s[2]), fmt)
        new_sequence += '[' + str(s[0]) + ',' + str((curr_time - prev_time).total_seconds()) + '] '
        prev_time = curr_time
    return new_sequence


def generate_train_test(sequence_file, n=None, ratio=0.3):
    anomaly_list = ['63a0d960-70b6-44c6-b606-491478a5cadf', 'd54b44eb-2d1a-4aa2-ba6b-074d35f8f12c']

    seq = pd.read_csv(sequence_file)
    seq['EventSequence'] = seq['EventSequence'].apply(compute_time_diff)
    seq["Label"] = seq["EventId"].apply(lambda x: 1 if x in anomaly_list else 0) #add label to the sequence of each blockid
    #seq["EventSequence"] = seq["EventSequence"].apply(lambda x: " ".join([str(i) for i in eval(x)]) if len(eval(x)) > 0 else None)
    #seq["EventSequence"].dropna(inplace=True)

    normal_seq = seq[seq["Label"] == 0]["EventSequence"]
    normal_seq = normal_seq.sample(frac=1, random_state=20) # shuffle normal data

    abnormal_seq = seq[seq["Label"] == 1]["EventSequence"]
    normal_len, abnormal_len = len(normal_seq), len(abnormal_seq)
    train_len = n if n else int(normal_len * ratio)
    print("normal size {0}, abnormal size {1}, train {2}".format(normal_len, abnormal_len, train_len))

    train = normal_seq.iloc[:train_len]
    test_normal = normal_seq.iloc[train_len:]
    test_abnormal = abnormal_seq

    train.to_csv(output_dir + "train.txt", index=False, header=False)
    test_normal.to_csv(output_dir + "test_normal.txt", index=False, header=False)
    test_abnormal.to_csv(output_dir + "test_abnormal.txt", index=False, header=False)
    print("generate train test data done")



if __name__ == "__main__":
    # 1. concat
    #concat_logs([abnormal_log_file, normal1_log_file, normal2_log_file], input_dir=input_dir, output_dir=output_dir)
    # 2. parser
    #3.mapping log keys to numbers
    #mapping()
    #sampling(log_structured_file)

    generate_train_test(log_sequence_file)
