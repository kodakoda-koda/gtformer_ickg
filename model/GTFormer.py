import torch
import torch.nn as nn

from layers.attention_free_layers import AFTFull
from layers.embed import TokenEmbedding_spatial, TokenEmbedding_temporal
from layers.self_attention import (
    Geospatial_SelfAttention,
    Relative_Temporal_SelfAttention,
    Spatial_SelfAttention,
    Temporal_SelfAttention,
)
from layers.transformer_encoder import Encoder, EncoderLayer


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.args = args

        # Temporal Transformer Block
        self.temporal_embedding = TokenEmbedding_temporal(args.num_tiles**2, args.d_model)

        if args.Temporal_mode == "BRPE":
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

    def forward(self, X, key_indices):
        # B: batch size
        # L: sequence length
        # O: num origin
        # D: num destination
        B, L, O, D = X.shape

        X = torch.cat([X, torch.zeros([B, 1, O, D]).to(self.device)], dim=1).reshape(B, L + 1, O * D)

        for _ in range(self.layers):
            temp_in = self.temporal_embedding(X)
            temp_out, A_temporal = self.temporal_transformer_encoder(temp_in, key_indices)
            temp_out = self.temporal_linear(temp_out)

            spat_in = self.spatial_embedding(X.permute(0, 2, 1))
            spat_out, A_spatial = self.spatial_transformer_encoder(spat_in, key_indices)
            spat_out = self.spatial_linear(spat_out)

            X = temp_out.reshape(B, L + 1, O, D) + spat_out.permute(0, 2, 1).reshape(B, L + 1, O, D)

        if self.args.save_outputs:
            return X[:, -1:, :, :], A_temporal, A_spatial
        else:
            return X[:, -1:, :, :]
