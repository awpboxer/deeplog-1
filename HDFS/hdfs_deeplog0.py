import argparse
import os
os.chdir("../")

from logdeep.models.lstm import deeplog, loganomaly, robustlog
from logdeep.tools.predict import Predicter
from logdeep.tools.train import Trainer
from logdeep.tools.utils import *

from logdeep.dataset.sample0 import sliding_window, session_window

# Config Parameters
options = dict()
options['data_dir'] = './output/hdfs/'
options['device'] = "cpu"


options["log_name"] = "hdfs"
#options["num_of_class"] = 24


# Smaple
options['sample'] = "session_window" #sliding window and session window(one line one session)
options['window_size'] = 10  #

# Features
options['sequentials'] = True
options['quantitatives'] = False  #one hot
options['semantics'] = False  #semantics vector like word embedding
options['feature_num'] = sum(
    [options['sequentials'], options['quantitatives'], options['semantics']])

# Model
options['input_size'] = 1
options['hidden_size'] = 64
options['num_layers'] = 2
options['num_classes'] = 28
options['embedding_size'] = 10
# Train
options['batch_size'] = 2048
options['accumulation_step'] = 1  #???

options['optimizer'] = 'adam'
options['lr'] = 0.001
options['max_epoch'] = 370
options['lr_step'] = (300, 350)  #多少轮更新一次学习率
options['lr_decay_ratio'] = 0.1

options['resume_path'] = None
options['model_name'] = "deeplog"
options['save_dir'] = options['data_dir'] + "result/"

# Predict
options['model_path'] = options['data_dir'] + "result/" + "deeplog_last.pth"
options['num_candidates'] = 9 # g candidates



seed_everything(seed=1234) #making results reproducible by seeding the RNG(random number generation)



def train():
    Model = deeplog(input_size=options['input_size'],
                    hidden_size=options['hidden_size'],
                    num_layers=options['num_layers'],
                    num_keys=options['num_classes'],
                    embedding_size=options['embedding_size'])
    trainer = Trainer(Model, options, train_normal, train_labels, test_normal, test_labels)
    trainer.start_train()


def predict():
    Model = deeplog(input_size=options['input_size'],
                    hidden_size=options['hidden_size'],
                    num_layers=options['num_layers'],
                    num_keys=options['num_classes'])
    predicter = Predicter(Model, options, test_normal, test_labels, test_abnormal, test_abnormal_labels)
    predicter.predict_unsupervised()


if __name__ == "__main__":
    print("Running deep log using HDFS")
    #train()
    predict()
    # parser = argparse.ArgumentParser()
    # parser.add_argument('mode', choices=['train', 'predict'])
    # args = parser.parse_args()
    # if args.mode == 'train':
    #     train()
    # else:
    #     predict()
