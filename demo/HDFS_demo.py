#!/usr/bin/env python


# get [log key, delta time] as input for deeplog
#log_structured_file = output_dir + log_file + "_structured.csv"
#log_templates_file = output_dir + log_file + "_templates.csv"

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


#################

