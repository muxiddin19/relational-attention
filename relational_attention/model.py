"""
Relational Transformer Model

Complete implementation of the Relational Transformer architecture,
including encoder, decoder, and type-constrained decoding.

Note: This implementation aims for clarity and correctness. For production use,
consider optimizing for efficiency (e.g., fused kernels, flash attention).

Tensor Shape Conventions:
- B: batch size
- S: source sequence length
- T: target sequence length
- D: hidden dimension
- K: number of attributes
- A: attribute dimension (D // K)
- H: number of attention heads
- V: vocabulary size
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List, Union
from dataclasses import dataclass

from .layers import (
    TupleEmbedding,
    SchemaAwarePositionalEncoding,
    RelationalTransformerBlock,
    RelationalTransformerEncoderBlock
)


@dataclass
class RelationalTransformerConfig:
    """Configuration for Relational Transformer models."""
    vocab_size: int = 32000
    hidden_dim: int = 768
    num_encoder_layers: int = 12
    num_decoder_layers: int = 12
    num_heads: int = 12
    num_attributes: int = 8
    ffn_dim: Optional[int] = None  # Defaults to 4 * hidden_dim
    max_seq_len: int = 2048
    dropout: float = 0.1
    padding_idx: int = 0
    tie_embeddings: bool = True

    # Attribute configuration
    # Attributes can be: 'learned' (end-to-end), 'fixed' (fixed projections)
    attribute_mode: str = 'learned'

    # Join attention configuration
    join_temperature: float = 1.0
    learnable_temperature: bool = True


class RelationalTransformerEncoder(nn.Module):
    """
    Relational Transformer Encoder.

    Stack of Relational Transformer blocks for encoding input sequences.
    Each token is represented as a tuple of K attributes, enabling
    attribute-specific attention patterns.

    Args:
        config: Model configuration

    Input Shapes:
        input_ids: (B, S) - Token indices
        attention_mask: (B, S) - 1 for real tokens, 0 for padding
        relation_ids: (B, S) - Optional relation/table membership IDs
        column_ids: (B, S) - Optional column type IDs

    Output Shapes:
        hidden_states: (B, S, D) - Encoded representations
        attention_weights: List of (H, B, S, S) per layer if return_attention=True
    """

    def __init__(self, config: RelationalTransformerConfig):
        super().__init__()

        self.config = config
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_encoder_layers

        # Tuple Embedding: tokens -> tuples with K attributes
        self.embedding = TupleEmbedding(
            vocab_size=config.vocab_size,
            hidden_dim=config.hidden_dim,
            num_attributes=config.num_attributes,
            padding_idx=config.padding_idx,
            dropout=config.dropout
        )

        # Schema-Aware Positional Encoding
        self.positional_encoding = SchemaAwarePositionalEncoding(
            hidden_dim=config.hidden_dim,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout
        )

        # Stack of Relational Transformer blocks
        self.layers = nn.ModuleList([
            RelationalTransformerBlock(
                hidden_dim=config.hidden_dim,
                num_heads=config.num_heads,
                num_attributes=config.num_attributes,
                ffn_dim=config.ffn_dim,
                dropout=config.dropout
            )
            for _ in range(config.num_encoder_layers)
        ])

        # Final layer norm
        self.final_norm = nn.LayerNorm(config.hidden_dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        relation_ids: Optional[torch.Tensor] = None,
        column_ids: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Encode input sequence.

        Args:
            input_ids: (B, S) Token indices
            attention_mask: (B, S) Attention mask (1=attend, 0=mask)
            relation_ids: (B, S) Optional relation IDs for schema encoding
            column_ids: (B, S) Optional column IDs for schema encoding
            return_attention: Whether to return attention weights

        Returns:
            Tuple of:
                - hidden_states: (B, S, D) Encoded representations
                - attention_weights: List[(H, B, S, S)] or None
        """
        # Embed tokens as tuples: (B, S) -> (B, S, D)
        # where D = K * A (K attributes of dimension A each)
        x = self.embedding(input_ids)

        # Add positional encoding: (B, S, D) -> (B, S, D)
        x = self.positional_encoding(x, relation_ids, column_ids)

        # Prepare attention mask: (B, S) -> (B, 1, 1, S)
        # This broadcasts over heads and query positions
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Apply transformer layers
        attention_weights = [] if return_attention else None
        for layer in self.layers:
            x, attn = layer(x, attention_mask, return_attention)
            if return_attention and attn is not None:
                attention_weights.append(attn)

        # Final normalization: (B, S, D) -> (B, S, D)
        x = self.final_norm(x)

        return x, attention_weights


class RelationalTransformerDecoder(nn.Module):
    """
    Relational Transformer Decoder.

    Stack of decoder blocks with self-attention and cross-attention,
    using Relational Attention for both.

    Args:
        config: Model configuration

    Input Shapes:
        input_ids: (B, T) - Target token indices
        encoder_output: (B, S, D) - Encoder hidden states
        self_attention_mask: (T, T) - Causal mask
        cross_attention_mask: (B, S) - Encoder attention mask

    Output Shape:
        hidden_states: (B, T, D)
    """

    def __init__(self, config: RelationalTransformerConfig):
        super().__init__()

        self.config = config
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_decoder_layers

        # Tuple Embedding
        self.embedding = TupleEmbedding(
            vocab_size=config.vocab_size,
            hidden_dim=config.hidden_dim,
            num_attributes=config.num_attributes,
            padding_idx=config.padding_idx,
            dropout=config.dropout
        )

        # Positional Encoding
        self.positional_encoding = SchemaAwarePositionalEncoding(
            hidden_dim=config.hidden_dim,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout
        )

        # Stack of decoder blocks
        self.layers = nn.ModuleList([
            RelationalTransformerEncoderBlock(
                hidden_dim=config.hidden_dim,
                num_heads=config.num_heads,
                num_attributes=config.num_attributes,
                ffn_dim=config.ffn_dim,
                dropout=config.dropout
            )
            for _ in range(config.num_decoder_layers)
        ])

        # Final layer norm
        self.final_norm = nn.LayerNorm(config.hidden_dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_output: torch.Tensor,
        self_attention_mask: Optional[torch.Tensor] = None,
        cross_attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Decode with cross-attention to encoder output.

        Args:
            input_ids: (B, T) Target token indices
            encoder_output: (B, S, D) Encoder output
            self_attention_mask: (T, T) Causal mask for self-attention
            cross_attention_mask: (B, S) Mask for cross-attention

        Returns:
            hidden_states: (B, T, D) Decoded representations
        """
        # Embed tokens: (B, T) -> (B, T, D)
        x = self.embedding(input_ids)

        # Add positional encoding
        x = self.positional_encoding(x)

        # Prepare cross-attention mask if provided
        if cross_attention_mask is not None:
            cross_attention_mask = cross_attention_mask.unsqueeze(1).unsqueeze(2)

        # Apply decoder layers
        for layer in self.layers:
            x = layer(x, encoder_output, self_attention_mask, cross_attention_mask)

        # Final normalization
        x = self.final_norm(x)

        return x


class TypeConstrainedDecoder(nn.Module):
    """
    Type-Constrained Decoding for structured generation.

    Constrains the output distribution to schema-valid outputs
    by applying masks based on the current generation state.

    P(y_t | y_{<t}, x) = softmax(o_t + m_t)
    where m_t ∈ {0, -∞}^|V| masks invalid tokens.

    This is particularly useful for:
    - SQL generation (valid keywords at each position)
    - Code generation (syntactically valid tokens)
    - Structured output (JSON, XML fields)

    Args:
        hidden_dim: Model hidden dimension
        vocab_size: Size of vocabulary

    Input Shapes:
        hidden_states: (B, T, D)
        valid_token_mask: (B, T, V) or (B, V) or None

    Output Shape:
        logits: (B, T, V)
    """

    def __init__(self, hidden_dim: int, vocab_size: int):
        super().__init__()
        self.output_projection = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute type-constrained logits.

        Args:
            hidden_states: (B, T, D) Hidden states
            valid_token_mask: Boolean mask for valid tokens
                - (B, T, V): Per-position mask
                - (B, V): Broadcast across positions
                - None: No constraints

        Returns:
            logits: (B, T, V) Output logits
        """
        # Project to vocabulary: (B, T, D) -> (B, T, V)
        logits = self.output_projection(hidden_states)

        # Apply type constraints if provided
        if valid_token_mask is not None:
            # Expand mask if needed
            if valid_token_mask.dim() == 2:
                valid_token_mask = valid_token_mask.unsqueeze(1)

            # Mask invalid tokens with -inf
            logits = logits.masked_fill(~valid_token_mask, float('-inf'))

        return logits


class RelationalTransformer(nn.Module):
    """
    Complete Relational Transformer model.

    Encoder-decoder architecture with Relational Attention,
    suitable for sequence-to-sequence tasks like text-to-SQL,
    semantic parsing, and structured generation.

    The key innovation is treating token representations as tuples
    with K typed attributes and using attribute-specific join attention.

    Architecture:
        Encoder: Stack of RelationalTransformerBlocks
        Decoder: Stack of RelationalTransformerEncoderBlocks (with cross-attention)
        Output: TypeConstrainedDecoder for structured generation

    Args:
        config: RelationalTransformerConfig or dict of config options

    Example:
        >>> config = RelationalTransformerConfig(
        ...     vocab_size=32000,
        ...     hidden_dim=768,
        ...     num_encoder_layers=12,
        ...     num_decoder_layers=12,
        ...     num_heads=12,
        ...     num_attributes=8
        ... )
        >>> model = RelationalTransformer(config)
        >>> output = model(input_ids, decoder_input_ids)
        >>> logits = output['logits']  # (B, T, V)
    """

    def __init__(
        self,
        config: Union[RelationalTransformerConfig, dict]
    ):
        super().__init__()

        # Handle dict config
        if isinstance(config, dict):
            config = RelationalTransformerConfig(**config)

        self.config = config
        self.vocab_size = config.vocab_size
        self.hidden_dim = config.hidden_dim
        self.padding_idx = config.padding_idx

        # Encoder
        self.encoder = RelationalTransformerEncoder(config)

        # Decoder
        self.decoder = RelationalTransformerDecoder(config)

        # Type-constrained output
        self.output = TypeConstrainedDecoder(config.hidden_dim, config.vocab_size)

        # Tie embeddings if specified
        if config.tie_embeddings:
            self.decoder.embedding.token_embedding.weight = \
                self.encoder.embedding.token_embedding.weight

        # Initialize weights
        self._init_weights()

        # Log model size
        self._log_model_info()

    def _init_weights(self):
        """Initialize model weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
                if module.padding_idx is not None:
                    nn.init.zeros_(module.weight[module.padding_idx])
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _log_model_info(self):
        """Log model information."""
        num_params = sum(p.numel() for p in self.parameters())
        num_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        self.num_parameters = num_params
        self.num_trainable_parameters = num_trainable

    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Create causal attention mask for decoder.

        Args:
            seq_len: Sequence length
            device: Device to create mask on

        Returns:
            mask: (seq_len, seq_len) with 1 for allowed, 0 for blocked
        """
        # Lower triangular mask (1s for allowed positions, 0s for blocked)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask

    def forward(
        self,
        input_ids: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        relation_ids: Optional[torch.Tensor] = None,
        column_ids: Optional[torch.Tensor] = None,
        valid_token_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of Relational Transformer.

        Args:
            input_ids: (B, S) Source token indices
            decoder_input_ids: (B, T) Target token indices (shifted right)
            attention_mask: (B, S) Source attention mask
            decoder_attention_mask: (B, T) Target attention mask (usually None, causal is auto)
            relation_ids: (B, S) Relation IDs for schema-aware encoding
            column_ids: (B, S) Column IDs for schema-aware encoding
            valid_token_mask: (B, T, V) or (B, V) Mask for type-constrained decoding
            labels: (B, T) Target labels for computing loss

        Returns:
            Dict containing:
                - logits: (B, T, V) Output logits
                - loss: Scalar if labels provided
                - encoder_output: (B, S, D) if needed for generation
        """
        # Encode source: (B, S) -> (B, S, D)
        encoder_output, _ = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            relation_ids=relation_ids,
            column_ids=column_ids
        )

        # Create causal mask for decoder: (T, T)
        tgt_len = decoder_input_ids.size(1)
        causal_mask = self._create_causal_mask(tgt_len, decoder_input_ids.device)

        # Decode: (B, T) + (B, S, D) -> (B, T, D)
        decoder_output = self.decoder(
            input_ids=decoder_input_ids,
            encoder_output=encoder_output,
            self_attention_mask=causal_mask,
            cross_attention_mask=attention_mask
        )

        # Get logits with type constraints: (B, T, D) -> (B, T, V)
        logits = self.output(decoder_output, valid_token_mask)

        output = {
            'logits': logits,
            'encoder_output': encoder_output
        }

        # Compute loss if labels provided
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=self.padding_idx)
            # Flatten for loss: (B*T, V) vs (B*T,)
            loss = loss_fn(
                logits.view(-1, self.vocab_size),
                labels.view(-1)
            )
            output['loss'] = loss

        return output

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 512,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        eos_token_id: int = 2,
        bos_token_id: int = 1,
        pad_token_id: int = 0,
        attention_mask: Optional[torch.Tensor] = None,
        relation_ids: Optional[torch.Tensor] = None,
        column_ids: Optional[torch.Tensor] = None,
        valid_token_mask: Optional[torch.Tensor] = None,
        do_sample: bool = True
    ) -> torch.Tensor:
        """
        Generate sequences autoregressively.

        Args:
            input_ids: (B, S) Source token indices
            max_length: Maximum generation length
            temperature: Sampling temperature (1.0 = no change)
            top_k: Top-k sampling (0 = disabled)
            top_p: Nucleus sampling threshold (1.0 = disabled)
            eos_token_id: End of sequence token ID
            bos_token_id: Beginning of sequence token ID
            pad_token_id: Padding token ID
            attention_mask: (B, S) Source attention mask
            relation_ids: (B, S) Relation IDs
            column_ids: (B, S) Column IDs
            valid_token_mask: (B, V) Valid token mask
            do_sample: Whether to sample (False = greedy)

        Returns:
            generated_ids: (B, gen_len) Generated token indices
        """
        batch_size = input_ids.size(0)
        device = input_ids.device

        # Encode source once
        encoder_output, _ = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            relation_ids=relation_ids,
            column_ids=column_ids
        )

        # Initialize decoder input with BOS token
        decoder_input = torch.full(
            (batch_size, 1),
            bos_token_id,
            dtype=torch.long,
            device=device
        )

        # Track which sequences have finished
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # Generate tokens autoregressively
        for step in range(max_length - 1):
            # Create causal mask
            tgt_len = decoder_input.size(1)
            causal_mask = self._create_causal_mask(tgt_len, device)

            # Decode
            decoder_output = self.decoder(
                input_ids=decoder_input,
                encoder_output=encoder_output,
                self_attention_mask=causal_mask,
                cross_attention_mask=attention_mask
            )

            # Get logits for last position: (B, 1, V) -> (B, V)
            logits = self.output(decoder_output[:, -1:], valid_token_mask)
            logits = logits.squeeze(1)  # (B, V)

            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature

            # Apply top-k filtering
            if top_k > 0:
                top_k_values = torch.topk(logits, min(top_k, logits.size(-1)))[0]
                threshold = top_k_values[:, -1].unsqueeze(-1)
                logits = logits.masked_fill(logits < threshold, float('-inf'))

            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )

                # Remove tokens with cumulative prob > top_p
                sorted_mask = cumulative_probs > top_p
                sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
                sorted_mask[:, 0] = False

                # Scatter back
                mask = sorted_mask.scatter(1, sorted_indices, sorted_mask)
                logits = logits.masked_fill(mask, float('-inf'))

            # Sample or greedy
            if do_sample:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)

            # Replace finished sequences' tokens with pad
            next_token = next_token.masked_fill(finished.unsqueeze(-1), pad_token_id)

            # Append to decoder input
            decoder_input = torch.cat([decoder_input, next_token], dim=1)

            # Update finished status
            finished = finished | (next_token.squeeze(-1) == eos_token_id)

            # Early stopping if all finished
            if finished.all():
                break

        return decoder_input

    def get_num_parameters(self, trainable_only: bool = False) -> int:
        """Get number of parameters."""
        if trainable_only:
            return self.num_trainable_parameters
        return self.num_parameters


class RelationalTransformerForSequenceClassification(nn.Module):
    """
    Relational Transformer for sequence classification tasks.

    Encoder-only model with a classification head, suitable for
    tasks like sentiment analysis, NLI, or sequence labeling.

    Args:
        config: Model configuration
        num_classes: Number of output classes
    """

    def __init__(
        self,
        config: Union[RelationalTransformerConfig, dict],
        num_classes: int
    ):
        super().__init__()

        if isinstance(config, dict):
            config = RelationalTransformerConfig(**config)

        self.config = config
        self.num_classes = num_classes

        # Encoder
        self.encoder = RelationalTransformerEncoder(config)

        # Classification head (pool [CLS] -> classify)
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.Tanh(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, num_classes)
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        relation_ids: Optional[torch.Tensor] = None,
        column_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for classification.

        Args:
            input_ids: (B, S) Token indices
            attention_mask: (B, S) Attention mask
            relation_ids: (B, S) Relation IDs
            column_ids: (B, S) Column IDs
            labels: (B,) Classification labels

        Returns:
            Dict with logits and optional loss
        """
        # Encode: (B, S) -> (B, S, D)
        encoder_output, _ = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            relation_ids=relation_ids,
            column_ids=column_ids
        )

        # Pool using [CLS] token: (B, S, D) -> (B, D)
        pooled = encoder_output[:, 0]

        # Classify: (B, D) -> (B, num_classes)
        logits = self.classifier(pooled)

        output = {'logits': logits}

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            output['loss'] = loss_fn(logits, labels)

        return output
