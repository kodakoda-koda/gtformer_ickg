import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.embed import TokenEmbedding_spatial, TokenEmbedding_temporal
from layers.self_attention import (
    AFTFull,
    Geospatial_SelfAttention,
    Relative_Temporal_SelfAttention,
    Spatial_SelfAttention,
    Temporal_SelfAttention,
)


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu

    def forward(self, x, key_indices):
        new_x, A = self.attention(x, key_indices)
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), A


class Encoder(nn.Module):
    def __init__(self, enc_layers, norm_layer):
        super(Encoder, self).__init__()
        self.enc_layers = nn.ModuleList(enc_layers)
        self.norm = norm_layer

    def forward(self, x, key_indices):
        for enc_layer in self.enc_layers:
            x, A = enc_layer(x, key_indices)

        x = self.norm(x)

        return x, A


class GTFormer_block(nn.Module):
    def __init__(self, args):
        super(GTFormer_block, self).__init__()

        self.args = args

        # Temporal Transformer Block
        self.temporal_embedding = TokenEmbedding_temporal(args.num_tiles**2, args.d_model)

        if args.temporal_mode == "BRPE":
            temporal_selfattention = Relative_Temporal_SelfAttention(
                args.d_model, args.n_head, args.seq_len + 1, args.save_outputs
            )
        else:
            temporal_selfattention = Temporal_SelfAttention(args.d_model, args.n_head, args.save_outputs)

        temporal_encoder_layers = [
            EncoderLayer(
                attention=temporal_selfattention, d_model=args.d_model, d_ff=args.d_model * 4, dropout=args.dropout
            )
            for _ in range(args.temporal_num_layers)
        ]

        temporal_norm = nn.LayerNorm(args.d_model)
        self.temporal_transformer_encoder = Encoder(temporal_encoder_layers, temporal_norm)
        self.temporal_linear = nn.Linear(args.d_model, args.num_tiles**2)

        # Geospatial Transformer Block
        self.spatial_embedding = TokenEmbedding_spatial(args.seq_len + 1, args.d_model)

        if args.spatial_mode == "KVR":
            spatial_selfattention = Geospatial_SelfAttention(args.d_model, args.n_head, args.save_outputs)
        elif args.spatial_mode == "AFT":
            spatial_selfattention = AFTFull(args.num_tiles**2, args.d_model, args.n_head, args.save_outputs)
        else:
            spatial_selfattention = Spatial_SelfAttention(args.d_model, args.n_head, args.save_outputs)

        spatial_encoder_layers = [
            EncoderLayer(
                attention=spatial_selfattention, d_model=args.d_model, d_ff=args.d_model * 4, dropout=args.dropout
            )
            for _ in range(args.spatial_num_layers)
        ]

        spatial_norm = nn.LayerNorm(args.d_model)
        self.spatial_transformer_encoder = Encoder(spatial_encoder_layers, spatial_norm)
        self.spatial_linear = nn.Linear(args.d_model, args.seq_len + 1)
        self.norm = nn.LayerNorm(args.num_tile**2)

    def forward(self, X, key_indices):
        temp_in = self.temporal_embedding(X)
        temp_out, A_temporal = self.temporal_transformer_encoder(temp_in, key_indices)
        temp_out = self.temporal_linear(temp_out)

        spat_in = self.spatial_embedding(X.permute(0, 2, 1))
        spat_out, A_spatial = self.spatial_transformer_encoder(spat_in, key_indices)
        spat_out = self.spatial_linear(spat_out)

        X = temp_out + spat_out.permute(0, 2, 1)
        X = self.norm(X)

        return X, A_temporal, A_spatial
