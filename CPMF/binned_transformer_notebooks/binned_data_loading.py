import pandas as pd
import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torch.nn import TransformerEncoder
import ffn
import numpy as np


def get_log_returns(tick, start='2020-01-01', end='2022-01-01'):
    """Get closings and log returns"""
    closings = ffn.get(tick, start=start, end=end)
    returns = closings.to_log_returns().dropna()

    # lower_bin = -5
    # upper_bin = 5
    # bin_range = upper_bin-lower_bin+2
    # returns['log_returns'] = pd.cut(returns[tick], bins=np.array([-np.inf] + list(range(lower_bin, upper_bin+1)) + [np.inf])/100.0, labels=range(bin_range))
    return returns


def get_rolling_window(df:pd.DataFrame, context_length=None, predict_length=None, win_size=None, format='list'):#, return_dict=False):
    """Window Splitting"""
    if win_size is None:
        win_size = context_length + predict_length
    rolling_bins = df.rolling(window=(win_size), closed='left')

    if format=='dataframe':
        return [s for s in rolling_bins][win_size:]

    if format=='dict':#return_dict:
        return [s.to_dict(orient='list') for s in (rolling_bins)][win_size:]
    
    rolling_bins = [list(s.values) for s in (rolling_bins)][win_size:]
    # rolling_bins

    #data_df = pd.DataFrame({'x':rolling_bins})
    return rolling_bins#data_df
    

def bin_returns(df_col, lower_bin, upper_bin, num_bins, as_pct=True, get_label_map=False):#, bins=None):
    """Bin them"""
    denom = 100.0 if as_pct else 1
    bin_range = np.linspace(lower_bin, upper_bin, num_bins-1)
    bin_labels =range(num_bins)
    bin_vals = np.array([-np.inf] + list(bin_range) + [np.inf])/denom
    binned = pd.cut(df_col, bins=bin_vals, labels=bin_labels)
    return (binned, {bin_labels[i] : bin_vals[i] for i, _ in enumerate(bin_labels)}) if get_label_map else binned

