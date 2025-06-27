import torch
import torch.nn as nn
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func

class FlashAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by n_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # Reduce the projection dim to match desired output dim

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = dropout
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):

        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)  # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        context_vec = flash_attn_func(queries, keys, values, dropout_p=self.dropout, softmax_scale=None,
                        causal=True, window_size=(-1, -1), alibi_slopes=None, deterministic=False)
        """dropout_p should be set to 0.0 during evaluation
        Supports multi-query and grouped-query attention (MQA/GQA) by passing in KV with fewer heads
        than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
        For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
        0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.
        If window_size != (-1, -1), implements sliding window local attention. Query at position i
        will only attend to keys between
        [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.

        Arguments:
            q: (batch_size, seqlen, nheads, headdim)
            k: (batch_size, seqlen, nheads_k, headdim)
            v: (batch_size, seqlen, nheads_k, headdim)
            dropout_p: float. Dropout probability.
            softmax_scale: float. The scaling of QK^T before applying softmax.
                Default to 1 / sqrt(headdim).
            causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
            window_size: (left, right). If not (-1, -1), implements sliding window local attention.
            alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
                (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
                is added to the attention score of query i and key j.
            deterministic: bool. Whether to use the deterministic implementation of the backward pass,
                which is slightly slower and uses more memory. The forward pass is always deterministic.
        Return:
            out: (batch_size, seqlen, nheads, headdim).
        """
        context_vec = context_vec.reshape(b, num_tokens, self.d_out)
        return context_vec
