import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


def make_tgt_mask(size: int) -> torch.Tensor:
    return torch.where(torch.tril(torch.ones(size, size)) != 0, 0.0, float("-inf"))


def get_eos_sos_and_padding_vectors(dimensionality=512):
    sos = torch.ones((dimensionality))
    eos = torch.ones((dimensionality)) * 0.5
    padding = torch.zeros((dimensionality)) * 0.5
    return sos, eos, padding


def create_masks(
    encoded_text_targets : torch.Tensor,
    padded_cnn_output : torch.Tensor,
    padded_input_length : int,
    orig_input_length : int,
    padding_index : int,
):
    batch_size = padded_cnn_output.size(0)
    tgt_mask = make_tgt_mask(encoded_text_targets.size(0))
    memory_mask = torch.zeros([encoded_text_targets.size(0), padded_input_length]).type(
        torch.bool
    )
    allowed_values_mask = torch.zeros([batch_size, orig_input_length]).type(torch.bool)
    padding_size = padded_input_length - orig_input_length
    ignore_values_mask = torch.ones(
        (batch_size, padding_size)
    ).type(torch.bool)
    memory_padding_mask = torch.concat([allowed_values_mask, ignore_values_mask], dim=1)

    # If a BoolTensor is provided, the positions with the value of True will be ignored while the position with the value of False will be unchanged.
    tgt_padding_indices = (encoded_text_targets == padding_index).type(torch.bool)
    tgt_padding_mask = tgt_padding_indices.transpose(0, 1)

    return tgt_mask, memory_mask, tgt_padding_mask, memory_padding_mask


class TrDecoder(nn.Module):
    def __init__(self, dim_input, pos_encoding_max_length, num_classes):
        super().__init__()
        self.pos_embed = PositionalEncoding(dim_input, 0.0, pos_encoding_max_length)
        self.num_classes = num_classes

        decoder_layer = nn.TransformerDecoderLayer(d_model=dim_input, nhead=8)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)

    def convert_to_embedding(self, index_tensor):
        index_tensor_long = index_tensor.long()
        onehot_target_tokens = nn.functional.one_hot(
            index_tensor_long, num_classes=self.num_classes
        )
        return onehot_target_tokens

    def forward(
        self,
        memory,
        tgt,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        memory_with_pos = self.pos_embed(memory.permute(2, 0, 1))
        tgt_embedded = self.convert_to_embedding(tgt).cuda()
        tgt_with_pos = self.pos_embed(tgt_embedded)

        return self.transformer_decoder(
            tgt=tgt_with_pos,
            memory=memory_with_pos,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
