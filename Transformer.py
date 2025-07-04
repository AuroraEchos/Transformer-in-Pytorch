"""
Transformer in Pytorch.

Components needed:
    - Positional Encoding
    - Self-Attention
    - Multi-Head Attention
    - Feedforward Layer
    - Encoder Layer
    - Decoder Layer
    - Encoder
    - Decoder
    - Transformer Model

Date: 2025-07-04
Author: Wenhao Liu
"""

import torch
import math
from torch import Tensor
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """
    Implements the sinusoidal positional encoding for Transformer.

    Args:
        d_model: The dimension of the embedding vector.
        max_len: The maximum length of input sequences.

    Purpose:
        Add positional information to token embeddings so that
        the model can take sequence order into account.
    """
    def __init__(
            self,
            d_model: int,
            max_len: int=5000,
    ) -> None:
        super().__init__()

        # Create a matrix of [max_len, d_model] with positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)    # [max_len, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)    # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Add positional encoding to the input tensor.

        Args:
            x: Tensor of shape [batch_size, seq_len, d_model]

        Returns:
            Tensor of shape [batch_size, seq_len, d_model] with positional encoding added.
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return x
    
class ScaledDotProductAttention(nn.Module):
    """
    Implements scaled dot-product attention mechanism.

    Args:
        None
    
    Purpose:
        Computes attention weights and applies them to values.
        Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
    """
    
    def __init__(self) -> None:
        super().__init__()

    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            mask: Tensor=None
    ) -> Tensor:
        """
        Args:
            query: Tensor of shape [batch_size, n_heads, seq_len_q, d_k]
            key: Tensor of shape [batch_size, n_heads, seq_len_k, d_k]
            value: Tensor of shape [batch_size, n_heads, seq_len_v, d_v]
            mask: (Optional) Tensor of shape [batch_size, 1, 1, seq_len_k] or broadcastable

        Returns:
            output: Tensor of shape [batch_size, n_heads, seq_len_q, d_v]
            attn: Tensor of attention weights [batch_size, n_heads, seq_len_q, seq_len_k]
        """
        d_k = query.size(-1)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = torch.softmax(scores, dim=-1)

        output = torch.matmul(attn, value)

        return output, attn
    
class MultiHeadAttention(nn.Module):
    """
    Implements multi-head attention.

    Args:
        d_model: The dimension of input embedding.
        n_heads: The number of attention heads.

    Purpose:
        Allows the model to jointly attend to information
        from different representation subspaces at different positions.
    """
    def __init__(
            self,
            d_model: int,
            n_heads: int
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads

        # Define linear layers for q, k, v
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        # Output linear layer
        self.w_o = nn.Linear(d_model, d_model)

        # Scaled dot-product attention module
        self.attention = ScaledDotProductAttention()

    def forward(
            self, 
            query: Tensor, 
            key: Tensor, 
            value: Tensor, 
            mask: Tensor = None
    ) -> Tensor:
        """
        Args:
            query: Tensor of shape [batch_size, seq_len_q, d_model]
            key: Tensor of shape [batch_size, seq_len_k, d_model]
            value: Tensor of shape [batch_size, seq_len_v, d_model]
            mask: (Optional) Tensor for masking attention scores

        Returns:
            output: Tensor of shape [batch_size, seq_len_q, d_model]
        """
        batch_size = query.size(0)

        # Linear projections and split into heads
        q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)

        # Apply scaled dot-product attention
        out, attn = self.attention(q, k, v, mask)

        # Concatenate heads
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # Final linear projection
        output = self.w_o(out)

        return output, attn
    
class PositionwiseFeedForward(nn.Module):
    """
    Implements the position-wise feedforward layer.

    Args:
        d_model: The dimension of the input embedding.
        d_ff: The dimension of the hidden layer in the feedforward network.
        dropout: Dropout probability applied after each linear layer.

    Purpose:
        Applies two linear transformations with a ReLU activation in between,
        independently at each position.
    """
    def __init__(
            self,
            d_model: int,
            d_ff: int,
            dropout: float=0.1
    ) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, d_model]

        Returns:
            Tensor of shape [batch_size, seq_len, d_model]
        """
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class EncoderLayer(nn.Module):
    """
    Implements a single Transformer encoder layer.

    Args:
        d_model: The dimension of the input embedding.
        n_heads: The number of attention heads.
        d_ff: The dimension of the feedforward network.
        dropout: Dropout probability applied after attention and feedforward layers.

    Purpose:
        Consists of:
            - Multi-head self-attention with residual + layer norm
            - Position-wise feedforward network with residual + layer norm
    """
    def __init__(
            self,
            d_model: int,
            n_heads: int,
            d_ff: int,
            dropout: float=0.1
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(
            self, 
            x: Tensor, 
            mask: Tensor = None
    ) -> Tensor:
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, d_model]
            mask: (Optional) Mask tensor for attention

        Returns:
            Tensor of shape [batch_size, seq_len, d_model]
        """
        # Self-attention sub-layer
        attn_out, _ = self.self_attn(x, x, x, mask)
        x = x + self.dropout1(attn_out)
        x = self.norm1(x)

        # Feedforward sub-layer
        ff_out = self.feed_forward(x)
        x = x + self.dropout2(ff_out)
        x = self.norm2(x)

        return x

class DecoderLayer(nn.Module):
    """
    Implements a single Transformer decoder layer.

    Args:
        d_model: The dimension of the input embedding.
        n_heads: The number of attention heads.
        d_ff: The dimension of the feedforward network.
        dropout: Dropout probability applied after attention and feedforward layers.

    Purpose:
        Consists of:
            - Masked multi-head self-attention with residual + layer norm
            - Multi-head encoder-decoder attention with residual + layer norm
            - Position-wise feedforward network with residual + layer norm
    """
    def __init__(
            self, 
            d_model: int, 
            n_heads: int, 
            d_ff: int, 
            dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.cross_attn = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(
            self,
            x: Tensor,
            memory: Tensor,
            tgt_mask: Tensor = None,
            memory_mask: Tensor = None
    ) -> Tensor:
        """
        Args:
            x: Tensor of shape [batch_size, tgt_seq_len, d_model] (decoder input)
            memory: Tensor of shape [batch_size, src_seq_len, d_model] (encoder output)
            tgt_mask: (Optional) mask for decoder self-attention
            memory_mask: (Optional) mask for encoder-decoder attention

        Returns:
            Tensor of shape [batch_size, tgt_seq_len, d_model]
        """
        # Masked self-attention sub-layer
        self_attn_out, _ = self.self_attn(x, x, x, tgt_mask)
        x = x + self.dropout1(self_attn_out)
        x = self.norm1(x)

        # Encoder-decoder attention sub-layer
        cross_attn_out, _ = self.cross_attn(x, memory, memory, memory_mask)
        x = x + self.dropout2(cross_attn_out)
        x = self.norm2(x)

        # Feedforward sub-layer
        ff_out = self.feed_forward(x)
        x = x + self.dropout3(ff_out)
        x = self.norm3(x)

        return x
    
class Encoder(nn.Module):
    """
    Implements the Transformer encoder as a stack of encoder layers.

    Args:
        d_model: The dimension of the input embedding.
        n_heads: The number of attention heads.
        d_ff: The dimension of the feedforward network.
        num_layers: The number of encoder layers to stack.
        dropout: Dropout probability.

    Purpose:
        Applies positional encoding and a sequence of encoder layers.
    """
    def __init__(
            self, 
            d_model: int,
            n_heads: int,
            d_ff: int,
            num_layers: int,
            dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(
            self, 
            x: Tensor, 
            mask: Tensor = None
    ) -> Tensor:
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, d_model]
            mask: (Optional) mask tensor for padding

        Returns:
            Tensor of shape [batch_size, seq_len, d_model]
        """
        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm(x)
        return x

class Decoder(nn.Module):
    """
    Implements the Transformer decoder as a stack of decoder layers.

    Args:
        d_model: The dimension of the input embedding.
        n_heads: The number of attention heads.
        d_ff: The dimension of the feedforward network.
        num_layers: The number of decoder layers to stack.
        dropout: Dropout probability.

    Purpose:
        Applies positional encoding and a sequence of decoder layers.
    """
    def __init__(
            self, 
            d_model: int,
            n_heads: int,
            d_ff: int,
            num_layers: int,
            dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(
            self, 
            x: Tensor, 
            memory: Tensor, 
            tgt_mask: Tensor = None, 
            memory_mask: Tensor = None
    ) -> Tensor:
        """
        Args:
            x: Tensor of shape [batch_size, tgt_seq_len, d_model]
            memory: Tensor of shape [batch_size, src_seq_len, d_model] (encoder output)
            tgt_mask: (Optional) mask tensor for decoder self-attention
            memory_mask: (Optional) mask tensor for encoder-decoder attention

        Returns:
            Tensor of shape [batch_size, tgt_seq_len, d_model]
        """
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, memory_mask)
        x = self.norm(x)
        return x

class Transformer(nn.Module):
    """
    Implements the full Transformer model for sequence-to-sequence tasks.

    Args:
        src_vocab_size: Vocabulary size of the source language.
        tgt_vocab_size: Vocabulary size of the target language.
        d_model: Dimension of the embedding and model.
        n_heads: Number of attention heads.
        d_ff: Dimension of the feedforward network.
        num_encoder_layers: Number of layers in the encoder.
        num_decoder_layers: Number of layers in the decoder.
        dropout: Dropout probability.
        max_len: Maximum sequence length.

    Purpose:
        End-to-end Transformer with embeddings, positional encodings,
        encoder, decoder, and output projection layer.
    """
    def __init__(
            self, 
            src_vocab_size: int,
            tgt_vocab_size: int,
            d_model: int,
            n_heads: int,
            d_ff: int,
            num_encoder_layers: int,
            num_decoder_layers: int,
            dropout: float = 0.1,
            max_len: int = 5000
    ) -> None:
        super().__init__()

        # Embedding layers
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len)

        # Encoder and decoder
        self.encoder = Encoder(d_model, n_heads, d_ff, num_encoder_layers, dropout)
        self.decoder = Decoder(d_model, n_heads, d_ff, num_decoder_layers, dropout)

        # Final output projection layer
        self.output_layer = nn.Linear(d_model, tgt_vocab_size)

        # Dropout on embeddings
        self.dropout = nn.Dropout(dropout)

    def forward(
            self, 
            src: Tensor, 
            tgt: Tensor, 
            src_mask: Tensor = None, 
            tgt_mask: Tensor = None, 
            memory_mask: Tensor = None
    ) -> Tensor:
        """
        Args:
            src: Source input tensor of shape [batch_size, src_seq_len]
            tgt: Target input tensor of shape [batch_size, tgt_seq_len]
            src_mask: (Optional) Mask for source sequence
            tgt_mask: (Optional) Mask for target sequence
            memory_mask: (Optional) Mask for encoder-decoder attention

        Returns:
            Logits of shape [batch_size, tgt_seq_len, tgt_vocab_size]
        """
        # Embed and apply positional encoding to source
        src_emb = self.src_embedding(src) * math.sqrt(self.src_embedding.embedding_dim)
        src_emb = self.dropout(self.pos_encoding(src_emb))

        # Embed and apply positional encoding to target
        tgt_emb = self.tgt_embedding(tgt) * math.sqrt(self.tgt_embedding.embedding_dim)
        tgt_emb = self.dropout(self.pos_encoding(tgt_emb))

        # Encoder
        memory = self.encoder(src_emb, src_mask)

        # Decoder
        out = self.decoder(tgt_emb, memory, tgt_mask, memory_mask)

        # Final projection to vocab size
        logits = self.output_layer(out)

        return logits


# Test the Transformer model
if __name__ == "__main__":
    src_vocab_size = 1000
    tgt_vocab_size = 1000
    d_model = 512
    n_heads = 8
    d_ff = 2048
    num_encoder_layers = 6
    num_decoder_layers = 6
    dropout = 0.1
    max_len = 5000

    model = Transformer(
        src_vocab_size, tgt_vocab_size, d_model, n_heads,
        d_ff, num_encoder_layers, num_decoder_layers, dropout, max_len
    )

    print(model)


