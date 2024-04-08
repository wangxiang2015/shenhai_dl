import multiprocessing
import numpy as np
import pandas as pd
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from numpy import ndarray
from joblib import Parallel, delayed
from multiprocessing import Pool
from rich.progress import track
from tqdm import tqdm
from pathlib import Path

import wfdb
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from wfdb import processing

from scipy.io import loadmat
from scipy.signal import decimate, resample
from biosppy.signals import ecg
from biosppy.signals.tools import filter_signal

import utils
from utils import *
import settings

class ECGWindowAlignedDataset(Dataset):
    def __init__(self, df, window, nb_windows, src_path):
        ''' Return window length segments from ecg signal startig from random qrs peaks
            df: trn_df, val_df or tst_df
            window: ecg window length e.g 2500 (5 seconds)
            nb_windows: number of windows to sample from record
        '''
        self.df = df
        self.window = window
        self.nb_windows = nb_windows
        self.src_path = src_path

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get data
        row = self.df.iloc[idx]
        filename = str(self.src_path/(row.Patient + '.hea'))
        data, hdr = load_challenge_data(filename)
        seq_len = data.shape[-1] # get the length of the ecg sequence
        
        # Get top (normalized) features (excludes signal duration and demo feats)
        top_feats = utils.all_feats[all_feats.filename == row.Patient][feature_names[:nb_feats]].values
        # First, convert any infs to nans
        top_feats[np.isinf(top_feats)] = np.nan
        # Replace NaNs with feature means
        top_feats[np.isnan(top_feats)] = feat_means[None][np.isnan(top_feats)]
        # Normalize wide features
        feats_normalized = (top_feats - feat_means) / feat_stds
        # Use zeros (normalized mean) if cannot find patient features
        if not len(feats_normalized):
            feats_normalized = np.zeros(nb_feats)[None]
        
        # Apply band pass filter
        if filter_bandwidth is not None:
            data = apply_filter(data, filter_bandwidth)
        
        # Polarity check, per selected channel
        for ch_idx in polarity_check:
            try:
                # Get BioSPPy ECG object, using specified channel
                ecg_object = ecg.ecg(signal=data[ch_idx], sampling_rate=fs, show=False)

                # Get rpeaks and beat templates
                rpeaks = ecg_object['rpeaks']
                templates, rpeaks = extract_templates(data[ch_idx], rpeaks)

                # Polarity check (based on extremes of median templates)
                templates_min = np.min(np.median(templates, axis=1)) 
                templates_max = np.max(np.median(templates, axis=1))

                if np.abs(templates_min) > np.abs(templates_max):
                    # Flip polarity
                    data[ch_idx] *= -1
                    templates *= -1
            except:
                continue

        # Detect qrs complexes (use lead II)
        xqrs = processing.XQRS(data[1,:], fs=500.)
        xqrs.detect(verbose=False)
        
        data = normalize(data)
        lbl = row[classes].values.astype(int)
        
        qrs = xqrs.qrs_inds[xqrs.qrs_inds < seq_len - self.window] # keep qrs complexes that allow a full window
        
        # Window too large, adjust sequence with padding
        if not len(qrs):
            # Add just enough padding to allow qrs find
            pad = np.abs(np.min(seq_len - window, 0)) + xqrs.qrs_inds[0] + 1
            data = np.pad(data, ((0,0),(0,pad)))
            seq_len = data.shape[-1] # get the length of the ecg sequence
            qrs = xqrs.qrs_inds[xqrs.qrs_inds < seq_len - self.window] # keep qrs complexes that allow a full window
    
        starts = np.random.randint(len(qrs), size=self.nb_windows) # get start indices of ecg segment (from qrs complex)
        starts = qrs[starts]
        ecg_segs = np.array([data[:,start:start+self.window] for start in starts])
        return ecg_segs, feats_normalized, lbl, hdr, filename

class ECGWindowPaddingDataset(Dataset):
    def __init__(self, df, window, nb_windows, src_path):
        ''' Return randome window length segments from ecg signal, pad if window is too large
            df: trn_df, val_df or tst_df
            window: ecg window length e.g 2500 (5 seconds)
            nb_windows: number of windows to sample from record
        '''
        self.df = df
        self.window = window
        self.nb_windows = nb_windows
        self.src_path = src_path
        self.all_feats = settings.all_feats
        
        start_time = time.time()
        print(f"====== start_time: {start_time} ======== df_len:{len(df)}")
        
        
        #res = Parallel(n_jobs=3, backend='loky')(delayed(generate_input)(self.df.iloc[ii], self.src_path, self.window, self.nb_windows) for ii in track(range(len(self.df))))

        '''res = []
        for ii in track(range(len(self.df))):
            res.append(generate_input(self.df.iloc[ii], self.src_path, self.window, self.nb_windows))'''
        
        manager = multiprocessing.Manager()
        shared_vars = manager.dict({'df': self.df, 'all_feats': self.all_feats})

        _df_list = []
        for _ in range(len(self.df)):
            _df_list.append(self.df.copy())
        _all_feats = []
        for _ in range(len(self.df)):
            _all_feats.append(self.all_feats.copy())

        res = None
        with Pool(processes=30) as pool, tqdm(total=len(self.df)) as pbar:
            #res = [pool.apply_async(generate_input, (self.df.iloc[ii], self.src_path, self.window, self.nb_windows), callback=lambda _:pbar.update(1)) for ii in range(len(self.df))]
            res = [pool.apply_async(generate_input, (self.df.iloc[ii], self.src_path, self.window, self.nb_windows, _all_feats[ii]), callback=lambda _:pbar.update(1)) for ii in range(len(self.df))]
        res = [_.get() for _ in res]

        self.all_ecg_segs = np.array([_[0] for _ in res])
        self.all_feats_normalized = np.array([_[1] for _ in res])
        self.all_lbl = np.array([_[2] for _ in res])
        self.all_age_sex = np.array([_[3] for _ in res])
        self.all_filename = np.array([_[4] for _ in res])

        print(f"====== {time.time() - start_time} ============")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """# Get data
        row = self.df.iloc[idx]
        filename = str(self.src_path/(row.Patient + '.hea'))
        data, hea_info = load_challenge_data(filename)
        seq_len = data.shape[-1] # get the length of the ecg sequence
        
        # Get top (normalized) features (excludes signal duration and demo feats)
        top_feats = all_feats[all_feats.filename == row.Patient][feature_names[:nb_feats]].values
        # First, convert any infs to nans
        top_feats[np.isinf(top_feats)] = np.nan
        # Replace NaNs with feature means
        top_feats[np.isnan(top_feats)] = feat_means[None][np.isnan(top_feats)]
        # Normalize wide features
        feats_normalized = (top_feats - feat_means) / feat_stds
        # Use zeros (normalized mean) if cannot find patient features
        if not len(feats_normalized):
            feats_normalized = np.zeros(nb_feats)[None]
        
        # Apply band pass filter
        if filter_bandwidth is not None:
            data = apply_filter(data, filter_bandwidth)
            
        # Polarity check, per selected channel
        for ch_idx in polarity_check:
            try:
                # Get BioSPPy ECG object, using specified channel
                ecg_object = ecg.ecg(signal=data[ch_idx], sampling_rate=fs, show=False)

                # Get rpeaks and beat templates/complex
                rpeaks = ecg_object['rpeaks']
                templates, rpeaks = extract_templates(data[ch_idx], rpeaks)

                # Polarity check (based on extremes of median templates)
                templates_min = np.min(np.median(templates, axis=1))
                templates_max = np.max(np.median(templates, axis=1))

                if np.abs(templates_min) > np.abs(templates_max):
                    # Flip polarity
                    data[ch_idx] *= -1
                    templates *= -1
            except:
                continue
        
        data = normalize(data)
        lbl = row[classes].values.astype(int)
        
        # Add just enough padding to allow window
        pad = np.abs(np.min(seq_len - window, 0))
        if pad > 0:
            data = np.pad(data, ((0,0),(0,pad+1)))
            seq_len = data.shape[-1] # get the new length of the ecg sequence

        starts = np.random.randint(seq_len - self.window + 1, size=self.nb_windows) # get start indices of ecg segment
        ecg_segs = np.array([data[:,start:start+self.window] for start in starts])

        age:float = 57.0
        sex:float = 0.0
        for line in hea_info:
            if  'Age' in line:
                age = line.split(': ')[-1].strip()
                age = float(age) if float(age)>0 else 57.0
            if 'Sex' in line:
                sex = 1.0 if 'Female' in line else 0.0
        #return ecg_segs, feats_normalized, lbl, hea_info[:15], filename
        return ecg_segs, feats_normalized, lbl, np.array([age,sex]), filename"""

        return self.all_ecg_segs[idx], self.all_feats_normalized[idx], self.all_lbl[idx], self.all_age_sex[idx], self.all_filename[idx]

def load_challenge_data(header_file)->tuple[ndarray, list]:
    with open(header_file, 'r') as f:
        header = f.readlines()
    sampling_rate = int(header[0].split()[2])
    mat_file = header_file.replace('.hea', '.mat')
    x = loadmat(mat_file)
    recording = np.asarray(x['val'], dtype=np.float64)
    
    # Standardize sampling rate
    if sampling_rate > fs:
        recording = decimate(recording, int(sampling_rate / fs))
    elif sampling_rate < fs:
        recording = resample(recording, int(recording.shape[-1] * (fs / sampling_rate)), axis=1)
    
    return recording, header

def normalize(seq, smooth=1e-8):
    ''' Normalize each sequence between -1 and 1 '''
    return 2 * (seq - np.min(seq, axis=1)[None].T) / (np.max(seq, axis=1) - np.min(seq, axis=1) + smooth)[None].T - 1

def extract_templates(signal, rpeaks, before=0.2, after=0.4, fs=500):
    # convert delimiters to samples
    before = int(before * fs)
    after = int(after * fs)

    # Sort R-Peaks in ascending order
    rpeaks = np.sort(rpeaks)

    # Get number of sample points in waveform
    length = len(signal)

    # Create empty list for templates
    templates = []

    # Create empty list for new rpeaks that match templates dimension
    rpeaks_new = np.empty(0, dtype=int)

    # Loop through R-Peaks
    for rpeak in rpeaks:

        # Before R-Peak
        a = rpeak - before
        if a < 0:
            continue

        # After R-Peak
        b = rpeak + after
        if b > length:
            break

        # Append template list
        templates.append(signal[a:b])

        # Append new rpeaks list
        rpeaks_new = np.append(rpeaks_new, rpeak) 

    # Convert list to numpy array
    templates = np.array(templates).T

    return templates, rpeaks_new    

def apply_filter(signal, filter_bandwidth, fs=500):
        # Calculate filter order
        order = int(0.3 * fs)
        # Filter signal
        signal, _, _ = filter_signal(signal=signal, ftype='FIR', band='bandpass',
                                     order=order, frequency=filter_bandwidth, 
                                     sampling_rate=fs)
        return signal
        
def generate_input(used_df_row, src_path:Path, window, nb_windows, _all_feats):
    row = used_df_row
    filename = str(src_path/(row.Patient + '.hea'))
    data, hea_info = load_challenge_data(filename)
    seq_len = data.shape[-1] # get the length of the ecg sequence

    # Apply band pass filter
    if filter_bandwidth is not None:
        data = apply_filter(data, filter_bandwidth)
        
    # Polarity check, per selected channel
    for ch_idx in polarity_check:
        try:
            # Get BioSPPy ECG object, using specified channel
            ecg_object = ecg.ecg(signal=data[ch_idx], sampling_rate=fs, show=False)

            # Get rpeaks and beat templates/complex
            rpeaks = ecg_object['rpeaks']
            templates, rpeaks = extract_templates(data[ch_idx], rpeaks)

            # Polarity check (based on extremes of median templates)
            templates_min = np.min(np.median(templates, axis=1))
            templates_max = np.max(np.median(templates, axis=1))

            if np.abs(templates_min) > np.abs(templates_max):
                # Flip polarity
                data[ch_idx] *= -1
                templates *= -1
        except:
            continue
    
    data = normalize(data)

    # Add just enough padding to allow window
    pad = np.abs(np.min(seq_len - window, 0))
    if pad > 0:
        data = np.pad(data, ((0,0),(0,pad+1)))
        seq_len = data.shape[-1] # get the new length of the ecg sequence

    starts = np.random.randint(seq_len - window + 1, size=nb_windows) # get start indices of ecg segment
    ecg_segs = np.array([data[:,start:start+window] for start in starts])

    # Get top (normalized) features (excludes signal duration and demo feats)
    top_feats = _all_feats[_all_feats.filename == row.Patient][feature_names[:nb_feats]].values
    # First, convert any infs to nans
    top_feats[np.isinf(top_feats)] = np.nan
    # Replace NaNs with feature means
    top_feats[np.isnan(top_feats)] = feat_means[None][np.isnan(top_feats)]
    # Normalize wide features
    feats_normalized = (top_feats - feat_means) / feat_stds
    # Use zeros (normalized mean) if cannot find patient features
    if not len(feats_normalized):
        feats_normalized = np.zeros(nb_feats)[None]
    
    lbl = row[classes].values.astype(int)

    age:float = 57.0
    sex:float = 0.0
    for line in hea_info:
        if  'Age' in line:
            age = line.split(': ')[-1].strip()
            age = float(age) if float(age)>0 else 57.0
        if 'Sex' in line:
            sex = 1.0 if 'Female' in line else 0.0
    
    age_sex = np.array([age,sex])

    return ecg_segs, feats_normalized, lbl, age_sex, filename

