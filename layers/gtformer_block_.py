import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.embed import TokenEmbedding_spatial, TokenEmbedding_temporal
from layers.self_attention import (
    AFTKVR,
    AFTFull,
    AFTSimple,
    KVR_Spatial_SelfAttention,
    Relative_Temporal_SelfAttention,
    Spatial_SelfAttention,
    Temporal_SelfAttention,
)


class EncoderLayer(nn.Module):
    def __init__(self, spatial_attention, temporal_attention, d_model, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.spatial_attention = spatial_attention
        self.temporal_attention = temporal_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu

    def forward(self, x):
        new_x, A_spatial = self.spatial_attention(x.permute(0, 2, 1))
        x = x + self.dropout(new_x)

        x = self.norm1(x)

        new_x, A_temporal = self.temporal_attention(x.permute(0, 2, 1))
        x = x + self.dropout(new_x)

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y), A_spatial, A_temporal


class Encoder(nn.Module):
    def __init__(self, enc_layer, norm_layer):
        super(Encoder, self).__init__()
        self.enc_layer = enc_layer
        self.norm = norm_layer

    def forward(self, x):
        x, A_spatial, A_temporal = self.enc_layer(x)
        x = self.norm(x)

        return x, A_spatial, A_temporal


class GTFormer_block(nn.Module):
    def __init__(self, args):
        super(GTFormer_block, self).__init__()

        self.args = args

        self.embedding = TokenEmbedding_temporal(args.num_tiles**2, args.num_tiles**2)

        if args.temporal_mode == "BRPE":
            temporal_selfattention = Relative_Temporal_SelfAttention(
                args.num_tiles**2, args.num_tiles, args.seq_len + 1, args.save_attention
            )
        else:
            temporal_selfattention = Temporal_SelfAttention(args.num_tiles**2, args.num_tiles, args.save_attention)

        if args.spatial_mode == "AFT-KVR":
            spatial_selfattention = AFTKVR(args.num_tiles, args.seq_len, 1, args.save_attention)
        elif args.spatial_mode == "AFT-full":
            spatial_selfattention = AFTFull(args.num_tiles, args.seq_len, 1, args.save_attention)
        elif args.spatial_mode == "AFT-simple":
            spatial_selfattention = AFTSimple(args.num_tiles, args.seq_len, 1, args.save_attention)
        elif args.spatial_mode == "KVR":
            spatial_selfattention = KVR_Spatial_SelfAttention(args.num_tiles, args.seq_len, 1, args.save_attention)
        else:
            spatial_selfattention = Spatial_SelfAttention(args.seq_len, 1, args.save_attention)

        encoder_layer = EncoderLayer(
            spatial_selfattention, temporal_selfattention, args.num_tiles**2, (args.num_tiles**2) * 4, args.dropout
        )

        norm = nn.LayerNorm(args.num_tiles**2)
        self.transformer_encoder = Encoder(encoder_layer, norm)
        self.linear = nn.Linear(args.num_tiles**2, args.num_tiles**2)

    def forward(self, X):
        input = self.embedding(X)
        out, A_spatial, A_temporal = self.transformer_encoder(input)
        out = self.linear(out)

        return out, A_temporal, A_spatial
