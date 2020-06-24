
import os
import re
import json
import pandas as pd
from collections import OrderedDict
from HDFS import hdfs_const

print(os.getcwd())
os.chdir("../")
print(os.getcwd())

# get [log key, delta time] as input for deeplog
input_dir  = '../data/hdfs_loghub/HDFS_1/'
output_dir = './output/hdfs/'  # The output directory of parsing results
log_file   = 'HDFS.log'  # The input log file name

output_dir += hdfs_const.PARSER + "_result/"

log_structured_file = output_dir + log_file + "_structured.csv"
log_templates_file = output_dir + log_file + "_templates.csv"
log_sequence_file = output_dir + "hdfs_sequence.csv"

#log_structured_df = pd.read_csv(log_structured_file)
#hdfs_input = []
#prev_time = None
#fmt = "%y%m%d %H%M%S"
#for idx, row in log_structured_df.iterrows():
#    if idx == 0:
#        delta_time = 0
#        prev_time = datetime.strptime("{0} {1}".format(row["Date"],row["Time"]), fmt)
#    else:
#        curr_time = datetime.strptime("{0} {1}".format(row["Date"],row["Time"]), fmt)
#        delta_time = (curr_time - prev_time).total_seconds()
#        prev_time = curr_time
#
#    log_key = row["EventId"]
#    hdfs_input.append([log_key, delta_time])
# pd.DataFrame(hdfs_input).head(20)

# generate train, val, predict file
# test file path: data_dir += hdfs_result/hdfs_train.csv
# val file path: data_dir += hdfs_result/hdfs_test_normal.csv


# mapping eventid to number
def mapping():
    log_temp = pd.read_csv(log_templates_file)
    log_temp.sort_values(by = ["Occurrences"], ascending=False, inplace=True)
    log_temp_dict = {event: idx+1 for idx , event in enumerate(list(log_temp["EventId"])) }
    print(log_temp_dict)
    with open (output_dir + "hdfs_log_templates.json", "w") as f:
        json.dump(log_temp_dict, f)


def hdfs_sampling(log_file, window='session', window_size=0):
    assert window == 'session', "Only window=session is supported for HDFS dataset."
    print("Loading", log_file)
    struct_log = pd.read_csv(log_file, engine='c',
            na_filter=False, memory_map=True)

    with open(output_dir + "hdfs_log_templates.json", "r") as f:
        event_num = json.load(f)

    data_dict = OrderedDict() #preserve insertion order of items
    for idx, row in struct_log.iterrows():
        blkId_list = re.findall(r'(blk_-?\d+)', row['Content'])
        blkId_set = set(blkId_list)
        for blk_Id in blkId_set:
            if not blk_Id in data_dict:
                data_dict[blk_Id] = []
            num = event_num.get(row["EventId"])
            if num and num <= 28: # extract top 28 events with most occurrences
                data_dict[blk_Id].append(num)
            else:
                print("No mapping for EventId: %s"%row["EventId"])
                print(row)
    data_dict = {k: v for k, v in data_dict if len(v) > 0} # remove blockid without eventid in sequence
    data_df = pd.DataFrame(list(data_dict.items()), columns=['BlockId', 'EventSequence'])
    data_df.to_csv(log_sequence_file,index=None)
    print("hdfs sampling done")


def generate_train_test(hdfs_sequence_file, n=None, ratio=0.3 ):
    blk_label_dict = {}
    blk_label_file = os.path.join(input_dir, "anomaly_label.csv")
    blk_df = pd.read_csv(blk_label_file)
    for _ , row in blk_df.iterrows():
        blk_label_dict[row["BlockId"]] = 1 if row["Label"] == "Anomaly" else 0

    seq = pd.read_csv(hdfs_sequence_file)
    seq["Label"] = seq["BlockId"].apply(lambda x: blk_label_dict[x]) #add label to the sequence of each blockid
    seq["EventSequence"] = seq["EventSequence"].apply(lambda x: " ".join([str(i) for i in eval(x)]) if len(eval(x)) > 0 else None)
    seq["EventSequence"].dropna(inplace=True)

    normal_seq = seq[seq["Label"] == 0]["EventSequence"]
    normal_seq = normal_seq.sample(frac=1, random_state=20) # shuffle normal data

    abnormal_seq = seq[seq["Label"] == 1]["EventSequence"]
    normal_len, abnormal_len = len(normal_seq), len(abnormal_seq)
    train_len = n if n else int(normal_len * ratio)
    print("normal size {0}, abnormal size {1}, train {2}".format(normal_len, abnormal_len, train_len))

    train = normal_seq.iloc[:train_len]
    test_normal = normal_seq.iloc[train_len:]
    test_abnormal = abnormal_seq

    train.to_csv(output_dir + "/hdfs_train.txt", index=False, header=False)
    test_normal.to_csv(output_dir + "/hdfs_test_normal.txt", index=False, header=False)
    test_abnormal.to_csv(output_dir + "hdfs_test_abnormal.txt", index=False, header=False)
    print("generate train test data done")



if __name__ == "__main__":
    #mapping()
    #hdfs_sampling(log_structured_file)
    generate_train_test(log_sequence_file)
