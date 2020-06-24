# pipeline
# 1. read origin logs
# 2. extract label, time and origin event
# 3. match event to the template
import re
import pandas as pd
from tqdm import tqdm
from HDFS import hdfs_const

source_data_dir = "../../data/hdfs_loghub/HDFS_1/"
output_dir = "../output/hdfs/"
para = {"hdfs":source_data_dir + "HDFS.log",
        "template":f"{output_dir}/{hdfs_const.PARSER}_result/HDFS.log_templates.csv",
        "structured_file":f"{output_dir}/{hdfs_const.PARSER}_result/hdfs_structured.csv"}

# read origin logs
def data_read(filepath):
    fp = open(filepath, "r")
    datas = []
    lines = fp.readlines()
    i = 0
    for line in tqdm(lines):
        row = line.strip("\n").split()
        datas.append(row)
        i = i + 1
        if i % 1000 == 0:
            print("read %s lines"%i)
    fp.close()
    return datas

def match(HDFS):
    # match event to the template
    template = pd.read_csv(para["template"])

    event = []
    event2id = {}

    for i in range(template.shape[0]):
        event_id = template.iloc[i, template.columns.get_loc("EventId")]
        event_template = template.iloc[i, template.columns.get_loc("EventTemplate")]
        event2id[event_template] = event_id
        event.append(event_template)

    error_log = []
    eventmap = []
    print("Matching...")
    for log in tqdm(HDFS):
        log_event = " ".join(log[9:])
        for i,item in enumerate(event):
            if re.match(r''+item,log_event) and re.match(r''+item,log_event).span()[1] == len(log_event):
                eventmap.append(event2id[item])
                break
            if i == len(event)-1:
                eventmap.append('error')
                error_log.append(log_event)
    return eventmap
def structure(HDFS,eventmap):
    # extract label, time and origin event
    label = []
    time = []
    for log in tqdm(HDFS):
        label.append(log[0])
        time.append(log[4])

    HDFS_structured = pd.DataFrame(columns=["label","time","event_id"])
    HDFS_structured["label"] = label
    HDFS_structured["time"] = time
    HDFS_structured["event_id"] = eventmap
    # Remove logs which do not match the template(very few logs ......)
    HDFS_structured = HDFS_structured[(-HDFS_structured["event_id"].isin(["error"]))]
    HDFS_structured.to_csv(para["structured_file"],index=None)

if __name__ == "__main__":
    HDFS = data_read(para["HDFS"])
    eventmap = match(HDFS)
    structure(HDFS,eventmap)
