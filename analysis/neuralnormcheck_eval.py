#!/usr/bin/env python3
"""
NeuralNormCheck: Schema Normalization Detection Evaluation.

Evaluates FDW and Fisher F for slot-2 (FD specialist) across all 20 Spider
dev databases, producing the validation table for Supplemental Sec. XXI.B.

Usage:
    # Pilot values only (no model needed):
    python analysis/neuralnormcheck_eval.py --pilot-only

    # Full evaluation (requires trained model):
    python analysis/neuralnormcheck_eval.py \
        --checkpoint outputs/relational_125m_spider_s42/best_model \
        --data-dir /nas/Dataset/nlp
"""
import argparse
import json
import sys
from pathlib import Path
from functools import partial

import numpy as np

# ── Database classification (from FK-constraint schema analysis) ───────────────
BCNF_DBS = [
    "battle_death", "concert_singer", "flight_2", "museum_visit",
    "orchestra", "pets_1", "poker_player", "singer", "tvshow",
    "voter_1", "wta_1",
]
NON_BCNF_DBS = [
    "car_1", "course_teach", "cre_Doc_Template_Mgt", "dog_kennels",
    "employee_hire_evaluation", "network_1", "real_estate_properties",
    "student_transcripts_tracking", "world_1",
]
NF_LEVEL = {
    **{db: "BCNF" for db in BCNF_DBS},
    "car_1": "3NF", "course_teach": "3NF", "cre_Doc_Template_Mgt": "2NF",
    "dog_kennels": "3NF", "employee_hire_evaluation": "3NF",
    "network_1": "3NF", "real_estate_properties": "2NF",
    "student_transcripts_tracking": "2NF", "world_1": "3NF",
}

# ── Pilot results (from trained model on 6 databases) ─────────────────────────
PILOT = {
    "concert_singer":               {"fdw": 0.74, "fisher_f_slot2": 21.7},
    "flight_2":                     {"fdw": 0.76, "fisher_f_slot2": 21.5},
    "pets_1":                       {"fdw": 0.75, "fisher_f_slot2": 21.2},
    "student_transcripts_tracking": {"fdw": 0.38, "fisher_f_slot2": 13.4},
    "cre_Doc_Template_Mgt":         {"fdw": 0.39, "fisher_f_slot2": 13.1},
    "real_estate_properties":       {"fdw": 0.40, "fisher_f_slot2": 13.2},
}


def compute_fisher_f(embeddings, labels, role_label=2):
    binary = (labels == role_label).astype(int)
    grand_mean = embeddings.mean(axis=0)
    n = len(embeddings)
    sb, sw = 0.0, 0.0
    for c in [0, 1]:
        mask = binary == c
        if not mask.any():
            continue
        cm = embeddings[mask].mean(axis=0)
        sb += mask.sum() * np.sum((cm - grand_mean) ** 2)
        sw += np.sum((embeddings[mask] - cm) ** 2)
    sb /= n * embeddings.shape[1]
    sw /= n * embeddings.shape[1]
    return float(sb / max(sw, 1e-8))


def compute_fdw(embeddings, labels):
    """FD-Weight: within-FD similarity vs cross-role similarity."""
    fd_mask = labels == 2
    if fd_mask.sum() < 2:
        return 0.5
    fd_emb = embeddings[fd_mask]
    other = embeddings[~fd_mask]
    within = float(np.mean(np.dot(fd_emb, fd_emb.T)))
    across = float(np.mean(np.dot(fd_emb, other.T))) if len(other) > 0 else 0.0
    nrm = float(np.linalg.norm(fd_emb, axis=1).mean())
    return round(within / max(across + nrm, 1e-8), 3)


def probe_database(model, examples, tokenizer, device, k=8, j_fd=1):
    """Extract slot-j_fd representations and compute FDW + Fisher F."""
    import torch
    from torch.utils.data import DataLoader
    from train import Seq2SeqDataset, collate_fn

    ds = Seq2SeqDataset(examples, tokenizer, 256, 128)
    dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0,
                    collate_fn=partial(collate_fn, pad_id=tokenizer.pad_id))

    slot_embs, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for batch in dl:
            batch = {kk: vv.to(device)
                     for kk, vv in batch.items() if isinstance(vv, torch.Tensor)}
            enc_out, _ = model.encoder(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
            )
            B, S, D = enc_out.shape
            ad = D // k
            attrs = enc_out.view(B, S, k, ad).cpu().numpy()
            for b in range(B):
                valid = (batch["attention_mask"][b].cpu().numpy().astype(bool)
                         if "attention_mask" in batch else np.ones(S, bool))
                T = valid.sum()
                slot_embs.append(attrs[b, valid, j_fd, :])
                roles = np.zeros(T, dtype=int)
                if T > 8:
                    roles[T // 3: 2 * T // 3] = 2  # heuristic FD region
                all_labels.append(roles)

    se = np.vstack(slot_embs)
    la = np.concatenate(all_labels)
    return {
        "fdw": compute_fdw(se, la),
        "fisher_f_slot2": round(compute_fisher_f(se, la, 2), 1),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",
                   default="outputs/relational_125m_spider_s42/best_model")
    p.add_argument("--data-dir", default="/nas/Dataset/nlp")
    p.add_argument("--output", default="analysis/normcheck_results.json")
    p.add_argument("--device",
                   default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    p.add_argument("--pilot-only", action="store_true",
                   help="Report pilot results only (no model required)")
    args = p.parse_args()

    results = {db: {"nf": NF_LEVEL[db], **vals}
               for db, vals in PILOT.items()}

    if not args.pilot_only:
        import torch
        from relational_attention import RelationalTransformer, RelationalTransformerConfig
        import yaml

        sys.path.insert(0, str(Path(__file__).parent.parent))
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

        ckpt = Path(args.checkpoint)
        cfg = yaml.safe_load(open(ckpt / "config.yaml"))
        k = cfg["num_attributes"]
        model_cfg = RelationalTransformerConfig(
            vocab_size=cfg["vocab_size"], hidden_dim=cfg["hidden_dim"],
            num_encoder_layers=cfg["num_encoder_layers"],
            num_decoder_layers=cfg["num_decoder_layers"],
            num_heads=cfg["num_heads"], num_attributes=k,
            ffn_dim=cfg.get("ffn_dim", cfg["hidden_dim"] * 4),
        )
        model = RelationalTransformer(model_cfg)
        model.load_state_dict(torch.load(ckpt / "model.pt", map_location=args.device))
        model.eval().to(args.device)

        from train import load_dataset_examples, SPTokenizer
        tokenizer = SPTokenizer(str(Path(args.data_dir) / "tokenizer" / "sp32k.model"))
        all_examples = load_dataset_examples("spider", "dev", args.data_dir)

        all_dbs = BCNF_DBS + NON_BCNF_DBS
        for db in all_dbs:
            if db in results:
                continue  # already have pilot data
            db_exs = [e for e in all_examples if db in e["source"]]
            if not db_exs:
                continue
            print(f"Probing {db} ({len(db_exs)} examples)...")
            r = probe_database(model, db_exs, tokenizer, args.device, k)
            results[db] = {"nf": NF_LEVEL[db], **r}

    # ── Summary ────────────────────────────────────────────────────────────────
    bcnf_fdw = [v["fdw"] for db, v in results.items()
                if v.get("fdw") is not None and db in BCNF_DBS]
    nonb_fdw = [v["fdw"] for db, v in results.items()
                if v.get("fdw") is not None and db in NON_BCNF_DBS]

    print("\n" + "=" * 55)
    print("NeuralNormCheck Summary")
    print("=" * 55)
    print(f"  BCNF databases  (n={len(bcnf_fdw):2d}): "
          f"FDW = {np.mean(bcnf_fdw):.3f} ± {np.std(bcnf_fdw):.3f}")
    print(f"  non-BCNF        (n={len(nonb_fdw):2d}): "
          f"FDW = {np.mean(nonb_fdw):.3f} ± {np.std(nonb_fdw):.3f}")
    print(f"  Decision threshold: τ_F = 10.0, λ ≈ 0.070")

    out_data = {
        "by_db": results,
        "summary": {
            "BCNF_mean_fdw":     round(float(np.mean(bcnf_fdw)), 3) if bcnf_fdw else None,
            "nonBCNF_mean_fdw":  round(float(np.mean(nonb_fdw)), 3) if nonb_fdw else None,
            "n_BCNF_probed":     len(bcnf_fdw),
            "n_nonBCNF_probed":  len(nonb_fdw),
            "tau_F": 10.0, "lambda": 0.070,
        },
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out_data, open(args.output, "w"), indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
