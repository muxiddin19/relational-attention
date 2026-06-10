"""
RelCopy: Typed Copy Mechanism for Relational Attention.

Extends the standard Pointer-Generator (See et al., ACL 2017) with TYPED copy
attention aligned to RelAttn attribute slots:
  - Slot 0 (entity identity): entity names / proper nouns
  - Slot 1 (functional dependency): predicates / verbs
  - Slot 2 (schema structure): structural tokens

For general use (COGS, CFQ), a single-head copy gate is sufficient.
The typed variant activates automatically when num_copy_heads > 1.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class RelCopyGate(nn.Module):
    """Single-head copy gate (standard pointer-generator).

    At each decoding step t:
      copy_attn[b,t,s] = softmax_s(W_q h_dec @ W_k h_enc^T / sqrt(D))
      copy_dist[b,t,v] = sum_{s: src[b,s]=v} copy_attn[b,t,s]
      gate[b,t]        = sigmoid(W_g [h_dec; ctx])
      P_final          = gate * P_gen + (1 - gate) * P_copy
    """

    def __init__(self, hidden_dim: int, vocab_size: int):
        super().__init__()
        self.query_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key_proj   = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.gate_proj  = nn.Linear(hidden_dim * 2, 1)
        self.vocab_size = vocab_size
        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.xavier_uniform_(self.key_proj.weight)
        nn.init.xavier_uniform_(self.gate_proj.weight)
        nn.init.zeros_(self.gate_proj.bias)

    def forward(
        self,
        decoder_hidden: torch.Tensor,
        encoder_output: torch.Tensor,
        src_ids: torch.Tensor,
        gen_logits: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, D = decoder_hidden.shape
        S = encoder_output.size(1)
        V = self.vocab_size

        q = self.query_proj(decoder_hidden)
        k = self.key_proj(encoder_output)
        attn_logits = torch.bmm(q, k.transpose(1, 2)) * (D ** -0.5)
        if encoder_mask is not None:
            attn_logits = attn_logits.masked_fill(
                encoder_mask.unsqueeze(1) == 0, float("-inf")
            )
        copy_attn = F.softmax(attn_logits, dim=-1)

        copy_dist = torch.zeros(B, T, V, device=decoder_hidden.device,
                                dtype=decoder_hidden.dtype)
        src_expand = src_ids.unsqueeze(1).expand(B, T, S)
        copy_dist.scatter_add_(2, src_expand, copy_attn)

        ctx = torch.bmm(copy_attn, encoder_output)
        gate = torch.sigmoid(self.gate_proj(torch.cat([decoder_hidden, ctx], dim=-1)))
        gen_probs = F.softmax(gen_logits, dim=-1)
        final_probs = gate * gen_probs + (1.0 - gate) * copy_dist
        return torch.log(final_probs.clamp(min=1e-9))


class TypedRelCopyGate(nn.Module):
    """Multi-head typed copy gate aligned with RelAttn attribute slots.

    num_copy_heads heads, each specialising in one attribute type:
      head 0 -> entity identity   (RelAttn slot 0)
      head 1 -> functional dep.   (RelAttn slot 1)
      head 2 -> schema structure  (RelAttn slot 2)

    A learned router selects which head to use at each decoding step.
    Falls back to single-head when num_copy_heads == 1.
    """

    def __init__(self, hidden_dim: int, vocab_size: int, num_copy_heads: int = 3):
        super().__init__()
        self.num_heads = num_copy_heads
        self.heads = nn.ModuleList([
            RelCopyGate(hidden_dim, vocab_size)
            for _ in range(num_copy_heads)
        ])
        if num_copy_heads > 1:
            self.router = nn.Linear(hidden_dim, num_copy_heads)

    def forward(
        self,
        decoder_hidden: torch.Tensor,
        encoder_output: torch.Tensor,
        src_ids: torch.Tensor,
        gen_logits: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.num_heads == 1:
            return self.heads[0](decoder_hidden, encoder_output,
                                 src_ids, gen_logits, encoder_mask)

        head_weights = F.softmax(self.router(decoder_hidden), dim=-1)
        head_logprobs = torch.stack([
            h(decoder_hidden, encoder_output, src_ids, gen_logits, encoder_mask)
            for h in self.heads
        ], dim=-1)
        mixed = (head_logprobs.exp() * head_weights.unsqueeze(2)).sum(dim=-1)
        return torch.log(mixed.clamp(min=1e-9))
