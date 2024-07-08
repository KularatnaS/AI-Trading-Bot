import unittest
from config.config import get_config

import numpy as np
import torch
import torch.nn as nn

from dataset.dataset import CandleDataset, get_label
from torch.utils.data import DataLoader

import logging
LOGGER = logging.getLogger(__name__)


class Test_dataset(unittest.TestCase):

    def test_get_label_max_min_above_cp_hold(self):
        # GIVEN
        encoder_input = np.array([[1, 2, 3, 4], [2, 3, 4, 0]])
        future_candles = np.array([[0, 0.0001, 0.00001, 3], [0, 0.0002, 0.000001, 5]])  # max = 0.0002, min = 0.000001

        # WHEN
        label = get_label(encoder_input, future_candles)

        # THEN
        self.assertEqual(label, 2)  # hold

    def test_get_label_max_min_above_cp_buy(self):
        # GIVEN
        encoder_input = np.array([[1, 2, 3, 4], [2, 3, 4, 0]])
        future_candles = np.array([[4, 6, 2, 3], [4, 8, 1, 5]])  # max = 8, min = 1

        # WHEN
        label = get_label(encoder_input, future_candles)

        # THEN
        self.assertEqual(label, 0)  # buy

    def test_get_label_max_min_below_cp_hold(self):
        # GIVEN
        encoder_input = np.array([[1, 2, 3, 4], [2, 3, 4, 0.0003]])
        future_candles = np.array([[0, 0.0001, 0.00001, 3], [0, 0.0002, 0.000001, 5]])

        # WHEN
        label = get_label(encoder_input, future_candles)

        # THEN
        self.assertEqual(label, 2)  # hold

    def test_get_label_max_min_below_cp_sell(self):
        # GIVEN
        encoder_input = np.array([[1, 2, 3, 4], [2, 3, 4, 10]])
        future_candles = np.array([[4, 6, 2, 3], [4, 8, 1, 5]])  # max = 8, min = 1

        # WHEN
        label = get_label(encoder_input, future_candles)

        # THEN
        self.assertEqual(label, 1)  # sell

    def test_candle_dataset_dataloader(self):
        # GIVEN
        data_dir = '../data/val/'
        seq_len = 10
        batch_size = 3
        config = get_config()
        d_model = config['d_model']
        candle_dataset = CandleDataset(data_dir, seq_len, config['prediction_units'])
        dataloader = DataLoader(candle_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        # WHEN
        # get one batch
        for batch in dataloader:
            encoder_input = batch['encoder_input']
            # encoder_mask = batch['encoder_mask']
            label = batch['label']

            # THEN
            self.assertEqual(encoder_input.size(), (batch_size, seq_len, d_model))
            #self.assertEqual(encoder_mask.size(), (batch_size, 1, seq_len, seq_len))
            self.assertEqual(label.size(), torch.Size([batch_size]))


if __name__ == '__main__':
    unittest.main()
