#!/usr/bin/env bash
# Download all benchmark datasets to NAS storage.
# Usage: NAS_DIR=/nas/Dataset/nlp bash scripts/download_datasets.sh
#
# Downloads: Spider, COGS, SCAN, CFQ, GSM8K
# Requires: wget, python3 with datasets/pandas installed

set -e

NAS_DIR="${NAS_DIR:-/nas/Dataset/nlp}"
# Use conda env python if available, fall back to system python3
PYTHON="${PYTHON:-$(which python3)}"
for p in \
    "$HOME/anaconda3/envs/relattn/bin/python" \
    "$HOME/miniconda3/envs/relattn/bin/python" \
    "$(which python3)"; do
    [ -x "$p" ] && { PYTHON="$p"; break; }
done
echo "Using Python: $PYTHON"
mkdir -p "$NAS_DIR"

echo "=== Downloading datasets to $NAS_DIR ==="
df -h "$NAS_DIR"

# -------------------------------------------------------
# 1. Spider (text-to-SQL)
# -------------------------------------------------------
SPIDER_DIR="$NAS_DIR/spider"
if [ ! -d "$SPIDER_DIR/train_spider.json" ] && [ ! -f "$SPIDER_DIR/train_spider.json" ]; then
    echo "--- Downloading Spider ---"
    mkdir -p "$SPIDER_DIR"
    # Primary: HuggingFace datasets (downloads train/dev splits + tables)
    $PYTHON - <<'EOF'
from datasets import load_dataset
import json, os, shutil

nas = os.environ.get("NAS_DIR", "/nas/Dataset")
out = f"{nas}/spider"
os.makedirs(out, exist_ok=True)

print("  Loading Spider from HuggingFace...")
ds = load_dataset("spider", trust_remote_code=True)

for split in ["train", "validation"]:
    fname = "train_spider.json" if split == "train" else "dev.json"
    rows = [dict(r) for r in ds[split]]
    with open(f"{out}/{fname}", "w") as f:
        json.dump(rows, f, indent=2)
    print(f"  Saved {len(rows)} rows -> {out}/{fname}")

print("  Spider download complete.")
EOF
    # Download database files (tables.json + SQLite DBs)
    wget -q -O "$SPIDER_DIR/spider_data.zip" \
        "https://drive.usercontent.google.com/download?id=1iRDVHLr4mX2wQKSgA9OEtSZJqypKHrD&confirm=t" \
        || echo "  WARNING: Spider DB zip failed (need Google Drive access). Download manually."
    if [ -f "$SPIDER_DIR/spider_data.zip" ]; then
        unzip -q "$SPIDER_DIR/spider_data.zip" -d "$SPIDER_DIR" && rm "$SPIDER_DIR/spider_data.zip"
    fi
    # Clone official evaluation script
    if [ ! -d "$SPIDER_DIR/test-suite-sql-eval" ]; then
        git clone --quiet https://github.com/taoyds/test-suite-sql-eval.git "$SPIDER_DIR/test-suite-sql-eval"
    fi
    echo "  Spider done: $SPIDER_DIR"
else
    echo "  Spider already present, skipping."
fi

# -------------------------------------------------------
# 2. COGS (compositional generalization)
# -------------------------------------------------------
COGS_DIR="$NAS_DIR/cogs"
if [ ! -f "$COGS_DIR/train.tsv" ]; then
    echo "--- Downloading COGS ---"
    mkdir -p "$COGS_DIR"
    $PYTHON - <<'EOF'
from datasets import load_dataset
import os

nas = os.environ.get("NAS_DIR", "/nas/Dataset")
out = f"{nas}/cogs"
os.makedirs(out, exist_ok=True)

print("  Loading COGS from HuggingFace...")
ds = load_dataset("najoungkim/COGS", trust_remote_code=True)
for split in ds:
    rows = ds[split].to_pandas()
    fname = {"train": "train.tsv", "validation": "dev.tsv",
             "test": "test.tsv", "gen": "gen.tsv"}.get(split, f"{split}.tsv")
    rows.to_csv(f"{out}/{fname}", sep="\t", index=False)
    print(f"  Saved {len(rows)} rows -> {out}/{fname}")
print("  COGS done.")
EOF
    echo "  COGS done: $COGS_DIR"
else
    echo "  COGS already present, skipping."
fi

# -------------------------------------------------------
# 3. SCAN (systematic compositional generalization)
# -------------------------------------------------------
SCAN_DIR="$NAS_DIR/scan"
if [ ! -f "$SCAN_DIR/simple_train.json" ]; then
    echo "--- Downloading SCAN ---"
    mkdir -p "$SCAN_DIR"
    $PYTHON - <<'EOF'
from datasets import load_dataset
import json, os

nas = os.environ.get("NAS_DIR", "/nas/Dataset")
out = f"{nas}/scan"
os.makedirs(out, exist_ok=True)

for split_type in ["simple", "addprim_jump", "length"]:
    print(f"  Loading SCAN ({split_type})...")
    ds = load_dataset("scan", split_type)
    for split in ds:
        rows = [dict(r) for r in ds[split]]
        fname = f"{split_type}_{split}.json"
        with open(f"{out}/{fname}", "w") as f:
            json.dump(rows, f, indent=2)
        print(f"  Saved {len(rows)} rows -> {out}/{fname}")
print("  SCAN done.")
EOF
    echo "  SCAN done: $SCAN_DIR"
else
    echo "  SCAN already present, skipping."
fi

# -------------------------------------------------------
# 4. CFQ (compositional Freebase questions)
# -------------------------------------------------------
CFQ_DIR="$NAS_DIR/cfq"
if [ ! -f "$CFQ_DIR/mcd1_train.json" ]; then
    echo "--- Downloading CFQ ---"
    mkdir -p "$CFQ_DIR"
    $PYTHON - <<'EOF'
from datasets import load_dataset
import json, os

nas = os.environ.get("NAS_DIR", "/nas/Dataset")
out = f"{nas}/cfq"
os.makedirs(out, exist_ok=True)

for split_type in ["mcd1", "mcd2", "mcd3", "random_split"]:
    print(f"  Loading CFQ ({split_type})...")
    try:
        ds = load_dataset("cfq", split_type)
        for split in ds:
            rows = [dict(r) for r in ds[split]]
            fname = f"{split_type}_{split}.json"
            with open(f"{out}/{fname}", "w") as f:
                json.dump(rows, f, indent=2)
            print(f"  Saved {len(rows)} rows -> {out}/{fname}")
    except Exception as e:
        print(f"  WARNING: {split_type} failed: {e}")
print("  CFQ done.")
EOF
    echo "  CFQ done: $CFQ_DIR"
else
    echo "  CFQ already present, skipping."
fi

# -------------------------------------------------------
# 5. GSM8K (math word problems)
# -------------------------------------------------------
GSM8K_DIR="$NAS_DIR/gsm8k"
if [ ! -f "$GSM8K_DIR/train.json" ]; then
    echo "--- Downloading GSM8K ---"
    mkdir -p "$GSM8K_DIR"
    $PYTHON - <<'EOF'
from datasets import load_dataset
import json, os

nas = os.environ.get("NAS_DIR", "/nas/Dataset")
out = f"{nas}/gsm8k"
os.makedirs(out, exist_ok=True)

print("  Loading GSM8K from HuggingFace...")
ds = load_dataset("openai/gsm8k", "main")
for split in ds:
    rows = [dict(r) for r in ds[split]]
    fname = f"{split}.json"
    with open(f"{out}/{fname}", "w") as f:
        json.dump(rows, f, indent=2)
    print(f"  Saved {len(rows)} rows -> {out}/{fname}")
print("  GSM8K done.")
EOF
    echo "  GSM8K done: $GSM8K_DIR"
else
    echo "  GSM8K already present, skipping."
fi

echo ""
echo "=== All datasets ready in $NAS_DIR ==="
ls -lh "$NAS_DIR"
