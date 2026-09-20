import sys
import torch

sys.path.insert(0, ".")

from relational_attention.model import RelationalTransformerConfig, RelationalTransformerEncoder

vocab_size = 100
batch, seq_len = 2, 16

for strategy in ["cyclic", "random_fixed", "learned"]:
    cfg = RelationalTransformerConfig(
        vocab_size=vocab_size,
        hidden_dim=64,
        num_encoder_layers=2,
        num_heads=8,
        num_attributes=8,
        ffn_dim=128,
        dropout=0.1,
        use_gating=True,
        use_mixing=True,
        pairing_strategy=strategy,
        pairing_seed=1234,
    )
    model = RelationalTransformerEncoder(cfg)
    x = torch.randint(0, vocab_size, (batch, seq_len))
    out, _ = model(x)
    n_params = sum(p.numel() for p in model.parameters())
    head_pairs = model.layers[0].self_attention.head_pairs
    print(f"strategy={strategy:13s} out_shape={tuple(out.shape)} params={n_params} head_pairs={head_pairs}")

    loss = out.sum()
    loss.backward()
    any_grad = any(p.grad is not None for p in model.parameters())
    print(f"  backward OK, grads populated: {any_grad}")

print("ALL STRATEGIES OK")
