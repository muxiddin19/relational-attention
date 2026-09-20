import sys
import torch

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")

from relational_attention.model import RelationalTransformerConfig, RelationalTransformer

vocab_size = 100
batch, src_len, tgt_len = 2, 12, 10


def make_model(use_gating, use_mixing, pairing_strategy):
    cfg = RelationalTransformerConfig(
        vocab_size=vocab_size,
        hidden_dim=64,
        num_encoder_layers=2,
        num_decoder_layers=2,
        num_heads=8,
        num_attributes=8,
        ffn_dim=128,
        dropout=0.1,
        use_gating=use_gating,
        use_mixing=use_mixing,
        pairing_strategy=pairing_strategy,
        pairing_seed=1234,
    )
    return RelationalTransformer(cfg)


configs = [
    ("baseline",     True,  True,  "cyclic"),
    ("no_gating",    False, True,  "cyclic"),
    ("no_mixing",    True,  False, "cyclic"),
    ("random_fixed", True,  True,  "random_fixed"),
    ("learned",      True,  True,  "learned"),
]

results = {}
for name, ug, um, ps in configs:
    model = make_model(ug, um, ps)
    n_params = sum(p.numel() for p in model.parameters())
    dec_head_pairs = model.decoder.layers[0].self_attention.head_pairs
    dec_cross_gating = model.decoder.layers[0].cross_attention.use_gating if hasattr(model.decoder.layers[0].cross_attention, "use_gating") else "n/a"

    src = torch.randint(0, vocab_size, (batch, src_len))
    tgt = torch.randint(0, vocab_size, (batch, tgt_len))
    enc_out, _ = model.encoder(src)
    dec_out = model.decoder(tgt, enc_out)
    logits = model.output(dec_out)
    loss = logits.sum()
    loss.backward()

    results[name] = n_params
    print(f"{name:13s} params={n_params:8d} dec_head_pairs={dec_head_pairs} logits_shape={tuple(logits.shape)}")

print("\nparam diffs vs baseline:")
for name in results:
    print(f"  {name:13s} diff={results[name] - results['baseline']:+d}")
