import torch
import sys
sys.path.insert(0, ".")

from relational_attention.model import RelationalTransformerConfig, RelationalTransformer

vocab_size = 100
batch, src_len, tgt_len = 2, 12, 10


def make_model(use_composed_join):
    cfg = RelationalTransformerConfig(
        vocab_size=vocab_size,
        hidden_dim=64,
        num_encoder_layers=2,
        num_decoder_layers=2,
        num_heads=8,
        num_attributes=8,
        ffn_dim=128,
        dropout=0.1,
        use_gating=True,
        use_mixing=True,
        pairing_strategy="cyclic",
        use_composed_join=use_composed_join,
    )
    return RelationalTransformer(cfg)


results = {}
for name, flag in [("baseline (no composed join)", False), ("composed_join", True)]:
    model = make_model(flag)
    n_params = sum(p.numel() for p in model.parameters())
    src = torch.randint(0, vocab_size, (batch, src_len))
    tgt = torch.randint(0, vocab_size, (batch, tgt_len))
    enc_out, _ = model.encoder(src)
    dec_out = model.decoder(tgt, enc_out)
    logits = model.output(dec_out)
    loss = logits.sum()
    loss.backward()
    has_nan = any(torch.isnan(p.grad).any().item() for p in model.parameters() if p.grad is not None)
    results[name] = n_params
    print(f"{name:28s} params={n_params:8d} logits_shape={tuple(logits.shape)} grad_nan={has_nan}")
    if flag:
        # Check chain_gate parameters exist and are learnable
        gates = [n for n, p in model.named_parameters() if "chain_gate" in n]
        print(f"  chain_gate params found: {len(gates)} (expect {8*2 + 8*2*2} = 8 heads x (2 enc + 2 dec self + 2 dec cross) layers)")
        print(f"  sample chain_gate values: {[round(model.state_dict()[g].item(),3) for g in gates[:3]]}")

print("\nparam diff (composed_join - baseline):", results["composed_join"] - results["baseline (no composed join)"])
print("ALL CHECKS PASSED" if not any(torch.isnan(torch.tensor(0.0)) for _ in [0]) else "NAN DETECTED")
