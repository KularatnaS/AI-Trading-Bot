import unittest

import numpy as np
import torch
import torch.nn as nn

from model.model import PositionalEncoding, LayerNormalization, FeedForwardBlock, MultiHeadAttentionBlock, \
    ResidualConnection, EncoderBlock, Encoder, ProjectionLayer, build_transformer

import logging
LOGGER = logging.getLogger(__name__)


class Test_llm_model(unittest.TestCase):

    def test_positional_encoding(self):
        # GIVEN
        d_model = 4
        seq_len = 512
        batch_size = 2
        dropout = 0.0
        positional_encoding = PositionalEncoding(d_model, seq_len, dropout)

        # WHEN
        x = torch.ones(batch_size, seq_len, d_model)
        output = positional_encoding(x)

        # THEN
        self.assertEqual(output.shape, (batch_size, seq_len, d_model))
        self.assertTrue(np.allclose(output[0], output[1]))

    def test_layer_normalization(self):
        # GIVEN
        d_model = 10
        seq_len = 5
        batch_size = 2
        eps = 10**-6
        layer_normalization = LayerNormalization(eps)

        # WHEN
        x = torch.rand(batch_size, seq_len, d_model)
        output = layer_normalization(x)

        # THEN
        self.assertEqual(output.shape, (batch_size, seq_len, d_model))

    def test_feed_forward_block(self):
        # GIVEN
        d_model = 10
        d_ff = 20
        dropout = 0.1
        seq_len = 5
        batch_size = 2
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)

        # WHEN
        x = torch.rand(batch_size, seq_len, d_model)
        output = feed_forward_block(x)

        # THEN
        self.assertEqual(output.shape, (batch_size, seq_len, d_model))

    def test_multi_head_attention_block(self):
        # GIVEN
        d_model = 256
        dropout = 0.1
        h = 8
        seq_len = 5
        batch_size = 2
        multi_head_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        x = torch.rand(batch_size, seq_len, d_model)

        # WHEN / THEN
        output = multi_head_attention_block(x, x, x, None)
        self.assertEqual(output.shape, (batch_size, seq_len, d_model))

        # WHEN / THEN
        mask = torch.zeros(batch_size, h, seq_len, seq_len)
        output = multi_head_attention_block(x, x, x, mask)
        self.assertEqual(output.shape, (batch_size, seq_len, d_model))

    def test_residual_connection(self):
        # GIVEN
        d_model = 256
        d_ff = 512
        h = 8
        seq_len = 5
        batch_size = 2
        dropout = 0.1
        residual_connection = ResidualConnection(dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        x = torch.rand(batch_size, seq_len, d_model)

        # WHEN / THEN
        output = residual_connection(x, feed_forward_block)
        self.assertEqual(output.shape, (batch_size, seq_len, d_model))

    def test_encoder_block(self):
        # GIVEN
        d_model = 256
        d_ff = 512
        h = 8
        seq_len = 5
        batch_size = 2
        dropout = 0.1
        multi_head_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(multi_head_attention_block, feed_forward_block, dropout)
        x = torch.rand(batch_size, seq_len, d_model)

        # WHEN / THEN
        output = encoder_block(x, None)
        self.assertEqual(output.shape, (batch_size, seq_len, d_model))

        # WHEN / THEN
        mask = torch.zeros(batch_size, h, seq_len, seq_len)
        output = encoder_block(x, mask)
        self.assertEqual(output.shape, (batch_size, seq_len, d_model))

    def test_encoder(self):
        # GIVEN
        d_model = 256
        d_ff = 512
        h = 8
        seq_len = 5
        batch_size = 2
        dropout = 0.1
        multi_head_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(multi_head_attention_block, feed_forward_block, dropout)
        encoder = Encoder(nn.ModuleList([encoder_block, encoder_block, encoder_block, encoder_block]))
        x = torch.rand(batch_size, seq_len, d_model)

        # WHEN / THEN
        output = encoder(x, None)
        self.assertEqual(output.shape, (batch_size, seq_len, d_model))

        # WHEN / THEN
        mask = torch.zeros(batch_size, h, seq_len, seq_len)
        output = encoder(x, mask)
        self.assertEqual(output.shape, (batch_size, seq_len, d_model))

    def test_projection_layer(self):
        # GIVEN
        d_model = 4
        batch_size = 2
        seq_len = 5
        n_trade_decisions = 3
        projection_layer = ProjectionLayer(d_model, seq_len, n_trade_decisions)
        x = torch.rand(batch_size, seq_len, d_model)

        # WHEN / THEN
        output = projection_layer(x)
        self.assertEqual(output.shape, (batch_size, n_trade_decisions))

    def test_build_transformer(self):
        # GIVEN
        src_seq_len = 512
        d_model = 4
        N = 6
        h = 1
        dropout = 0.1
        d_ff = 1024
        batch_size = 2
        n_trade_decisions = 3
        transformer = build_transformer(src_seq_len, d_model, N, h, dropout, d_ff, n_trade_decisions)
        x = torch.rand(batch_size, src_seq_len, d_model)
        mask = torch.zeros(2, h, src_seq_len, src_seq_len)

        # WHEN / THEN
        output_encode = transformer.encode(x, mask)
        self.assertEqual(output_encode.shape, (batch_size, src_seq_len, d_model))

        # WHEN / THEN
        output_project = transformer.project(output_encode)
        self.assertEqual(output_project.shape, (batch_size, n_trade_decisions))


if __name__ == '__main__':
    unittest.main()
