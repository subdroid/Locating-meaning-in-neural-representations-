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
    sent1 = random.sample(sent1,100)
    sent0 = random.sample(sent0,100)
    
    return sent0, sent1
    
    
def extract_paraphrase_data(data_loc):
    random.seed(1314)
    file_loc = os.path.join(data_loc,os.path.join("dataset","msr-paraphrase-corpus"))
    data_src = os.path.join(file_loc,"msr_paraphrase_train.txt")
    f_target = open(data_src,"r").read().split("\n")
    sent0, sent1 = file_read_paraphrase(f_target)
    return sent0, sent1

def extract_synthetic_data(data_loc):
    sent = []
    Sent0 = open(os.path.join(data_loc,"original_sentences"),"r").read().split("\n")
    Sent1 = open(os.path.join(data_loc,"synthetic_sentences"),"r").read().split("\n")
    for l_index in range(len(Sent0)):
        s0 = Sent0[l_index]
        s1 = Sent1[l_index]
        if len(s0.split())!=0 and len(s1.split())!=0:
            sent.append([s0,s1])
    return sent
