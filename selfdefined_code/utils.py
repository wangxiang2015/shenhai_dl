import argparse
import json
import math
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
#import wfdb
#from wfdb import processing

import torch
import torch.nn as nn
import torch.nn.functional as F

# Parameters
debug = False
patience = 10

gpu_list = [0,1]
device = torch.device(f'cuda:{gpu_list[0]}') if torch.cuda.is_available() else torch.device('cpu')
#device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#batch_size = 128 * torch.cuda.device_count()
batch_size = 128 * len(gpu_list)
nb_epoch = 100

window = 15*500
dropout_rate = 0.2
deepfeat_sz = 64
padding = 'zero' # 'zero', 'qrs', or 'none'
fs = 500
filter_bandwidth = [3, 45]
polarity_check = []
model_name = 'ecg_model'

# Transformer parameters
d_model = 256   # embedding size
nhead = 8       # number of heads
d_ff = 2048     # feed forward layer size
num_layers = 8  # number of encoding layers

do_train = True

ch_idx = 1
nb_demo = 2
nb_feats = 20
thrs_per_class = False
class_weights = None

classes = sorted(['270492004', '164889003', '164890007', '426627000', '713427006', 
                  '713426002', '445118002', '39732003', '164909002', '251146004', 
                  '698252002', '10370003', '284470004', '427172004', '164947007', 
                  '111975006', '164917005', '47665007', '59118001', '427393009', 
                  '426177001', '426783006', '427084000', '63593006', '164934002', 
                  '59931005', '17338001'])

char2dir = {
        'Q' : 'Training_2',
        'A' : 'Training_WFDB',
        'E' : 'WFDB',
        'S' : 'WFDB',
        'H' : 'WFDB',
        'I' : 'WFDB'
    }

# Load all features dataframe
#data_df = pd.read_csv('records_stratified_10_folds_v2.csv', index_col=0)
data_df = pd.read_csv('used_filename.csv', index_col=0)
#data_df = data_df[0:1000]

all_feats = pd.concat([pd.read_csv(f, index_col=0) for f in list(Path('feats/').glob(f'*/*all_feats_ch_{ch_idx}.zip'))])    

leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
lead2idx = dict(zip(leads, range(len(leads))))

dx_mapping_scored = pd.read_csv('eval/dx_mapping_scored.csv')
snomed2dx = dict(zip(dx_mapping_scored['SNOMED CT Code'].values, dx_mapping_scored['Dx']))

beta = 2
num_classes = len(classes)

weights_file = 'eval/weights.csv'
normal_class = '426783006'
normal_index = classes.index(normal_class)
normal_lbl = [0. if i != normal_index else 1. for i in range(num_classes)]
equivalent_classes = [['713427006', '59118001'], ['284470004', '63593006'], ['427172004', '17338001']]

# Get feature names in order of importance (remove duration and demo)
feature_names = list(np.load('top_feats.npy'))
feature_names.remove('full_waveform_duration')
feature_names.remove('Age')
feature_names.remove('Gender_Male')

# Compute top feature means and stds
# Get top feats (exclude signal duration)
feats = all_feats[feature_names[:nb_feats]].values

# First, convert any infs to nans
feats[np.isinf(feats)] = np.nan

# Store feature means and stds
feat_means = np.nanmean(feats, axis=0)
feat_stds = np.nanstd(feats, axis=0)

def get_age(hdrs):
    ''' Get list of ages as integers from list of hdrs '''
    hs = []
    for h in hdrs:
        res = re.search(r': (\d+)\n', h)
        if res is None:
            hs.append(0)
        else:
            hs.append(float(res.group(1)))
    return np.array(hs)


"""
self-defined functions
"""
def new_dir(target_dir):
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
        os.mkdir(target_dir)
    else:
        os.mkdir(target_dir)
    
    return target_dir


def write_json(data, json_file:str)->None:
    """
    Parameters
    ----------
    data: int, float, list or dict
    json_file: str
        "/filepath/filename.json"
    """
    assert isinstance(data, (int, float, list, dict)), "error: data type is wrong."
    if not os.path.exists(os.path.dirname(json_file)):
        os.makedirs(os.path.dirname(json_file))
    with open(json_file, 'w') as f:
        json.dump(data, f)

def load_json(json_file:str):
    with open(json_file, 'r') as f:
        res = json.load(f)
    return res



if __name__=='__main__':
    top_feats = all_feats[all_feats.filename == 'A0007'][feature_names[:nb_feats]].values
    # First, convert any infs to nans
    top_feats[np.isinf(top_feats)] = np.nan
    aa = feat_means[None][np.isnan(top_feats)]
    print(aa)

