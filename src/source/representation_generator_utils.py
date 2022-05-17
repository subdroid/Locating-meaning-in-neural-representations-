#!/usr/bin/env python3
import os
import random
import json
import pickle
import pandas as pd



def pickle_saver(folder_loc,model_name,rep_list):
    rep_name = model_name.upper()+".pkl"
    file_loc = os.path.join(folder_loc,rep_name)
    with open(file_loc, 'wb') as f:
        pickle.dump(rep_list, f)
    
def check_folder_exists(target_folder):
    if not os.path.exists(target_folder):
        os.mkdir(target_folder)


def file_read_paraphrase(data):
    sent1 = []
    sent0 = []
    for line in data:
        components = line.split("\t")
        if len(components)==5:
            cat = components[0]
            if cat=="0":
                sent0.append([components[-1],components[-2]])
            elif cat=="1":
                sent1.append([components[-1],components[-2]])       
    sent1 = random.sample(sent1,1000)
    sent0 = random.sample(sent0,1000)
    
    return sent0, sent1
    
    
def extract_paraphrase_data(data_loc):
    random.seed(1314)
    file_loc = os.path.join(data_loc,os.path.join("dataset","msr-paraphrase-corpus"))
    data_src = os.path.join(file_loc,"msr_paraphrase_train.txt")
    f_target = open(data_src,"r").read().split("\n")
    sent0, sent1 = file_read_paraphrase(f_target)
    return sent0, sent1
