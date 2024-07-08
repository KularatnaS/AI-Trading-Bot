import os
import glob
import pandas as pd
from einops import rearrange

import torch
from torch.utils.data import Dataset
from config.config import get_config
import numpy as np

import matplotlib.pyplot as plt

def plot_candle_prediction(gt_candles, pred_candles):
    # gt_candles = (n_candles_to_predict, 4), where columns are open, high, low, close
    # pred_candles = (n_candles_to_predict, 4), where columns are open, high, low, close
    # I want to plot the high, low and mid of gt_candles as a green line where high and low lines are dashed
    # I want to plot the high, low and mid of pred_candles as a red line where high and low lines are dashed
    # Finally save the plot to a file
    fig, ax = plt.subplots()
    # ax.plot(gt_candles[:, 0], 'g-', label='gt open')
    ax.plot(gt_candles[:, 1], 'b-', label='gt high')
    ax.plot(gt_candles[:, 2], 'b--', label='gt low')
    ax.plot(gt_candles[:, 3], 'g--', label='gt close')
    # gt_mid = (gt_candles[:, 1] + gt_candles[:, 2]) / 2
    # ax.plot(gt_mid, 'g-', label='gt mid')
    # ax.plot(pred_candles[:, 0], 'r-', label='pred open')
    ax.plot(pred_candles[:, 1], 'r-', label='pred high')
    ax.plot(pred_candles[:, 2], 'r--', label='pred low')
    # ax.plot(pred_candles[:, 3], 'r--', label='pred close')
    # pred_mid = (pred_candles[:, 1] + pred_candles[:, 2]) / 2
    # ax.plot(pred_mid, 'r--', label='pred mid')
    plt.legend()
    plt.show()


class CandleDataset(Dataset):
    def __init__(self, data_dir, seq_len, prediction_units, config):
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.prediction_units = prediction_units
        self.config = config
        self.instrument = config['instrument']

        # read csv file at data_dir
        self.data = pd.read_csv(os.path.join(data_dir, f'{self.instrument}.csv'))
        # get o,h,l,c as a numpy array
        self.data = self.data[['o', 'h', 'l', 'c']].values

    def __len__(self):
        return len(self.data) - self.seq_len - self.prediction_units

    def __getitem__(self, idx):
        encoder_input = self.data[idx: idx + self.seq_len]
        future_candles = self.data[idx + self.seq_len: idx + self.seq_len + self.prediction_units]
        label = get_label(encoder_input, future_candles, self.config)
        encoder_input = torch.tensor(encoder_input, dtype=torch.float32)

        # normalize encoder_input
        mean = encoder_input.mean()
        std = encoder_input.std()
        encoder_input_normalised = (encoder_input - mean) / std

        # (seq_len, d_model) -> (seq_len * d_model)
        encoder_input_normalised = rearrange(encoder_input_normalised, 'seq_len d_model -> (seq_len d_model)')

        return {
            'encoder_input': encoder_input_normalised,  # (seq_len,)
            # 'encoder_mask': causal_mask(encoder_input.size(0)),  # (1, seq_len, seq_len)
            'label': torch.tensor(label, dtype=torch.float32).long(),  # (seq_len,)
            'future_candles': future_candles,
            'original_encoder_input': encoder_input
        }


def get_label(encoder_input, future_candles, config):
    # returns 0 for buy, 1 for sell and 2 for hold
    label = 2  # initialise at hold

    SPREAD = config['spread']
    MIN_PROFIT = config['min_profit']
    # get closing price of the last candle
    CP = encoder_input[-1][-1]
    # get max and min of future candles
    MAX = max(future_candles[:, 1])
    MIN = min(future_candles[:, 2])
    ATR = calculate_atr(encoder_input)

    delta_profit = ATR * config['atr_exit_factor_profit'] + SPREAD
    delta_loss = ATR * config['atr_exit_factor_loss'] - SPREAD

    if delta_profit >= MIN_PROFIT:
        if (MAX > CP) and (MIN < CP):
            # Buy scenario
            Increase = MAX - CP
            Decrease = CP - MIN

            if Increase > delta_profit and Decrease < delta_loss:
                label = 0  # buy

            if Decrease > delta_profit and Increase < delta_loss:
                label = 1  # sell
        elif MAX > CP and MIN > CP:
            if (MAX - CP) > delta_profit:
                label = 0  # buy
        elif MAX < CP and MIN < CP:
            if (CP - MIN) > delta_profit:
                label = 1  # sell
    return label


def calculate_atr(past_candles, period=14):

    # use last 14 past candles to calculate average true range
    high = past_candles[-period:, 1]
    low = past_candles[-period:, 2]

    # calculate average true range
    ranges = high - low
    average_true_range = np.mean(ranges)
    return average_true_range