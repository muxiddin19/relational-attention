#!/usr/bin/env bash
# Download all benchmark datasets to NAS storage.
# Usage: NAS_DIR=/nas/Dataset/nlp bash scripts/download_datasets.sh
#
# Downloads: Spider, COGS, SCAN, CFQ, GSM8K
# Requires: wget, python3 with datasets/pandas installed

set -euo pipefail
# Continue past individual dataset failures
download_with_fallback() { "$@" || echo "  WARNING: command failed, continuing."; }

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
    # HuggingFace: try multiple known Spider dataset paths
    $PYTHON - <<'EOF'
from datasets import load_dataset
import json, os

nas = os.environ.get("NAS_DIR", "/nas/Dataset/nlp")
out = f"{nas}/spider"
os.makedirs(out, exist_ok=True)

# Try known HuggingFace dataset names in order
candidates = ["xlangai/spider", "spider-db/spider", "spider"]
ds = None
for name in candidates:
    try:
        print(f"  Trying HuggingFace dataset: {name}")
        ds = load_dataset(name)
        print(f"  Loaded from: {name}")
        break
    except Exception as e:
        print(f"  {name} failed: {e}")

if ds is None:
    print("  WARNING: Could not load Spider from HuggingFace. Skipping NL/SQL pairs.")
    print("  Please manually download from https://yale-nlp.github.io/spider/")
else:
    split_map = {"train": "train_spider.json", "validation": "dev.json", "test": "test.json"}
    for split in ds:
        fname = split_map.get(split, f"{split}.json")
        rows = [dict(r) for r in ds[split]]
        with open(f"{out}/{fname}", "w") as f:
            json.dump(rows, f, indent=2)
        print(f"  Saved {len(rows)} rows -> {out}/{fname}")
    print("  Spider NL/SQL pairs done.")
EOF
    # Download SQLite DB files via gdown (Google Drive)
    if [ ! -d "$SPIDER_DIR/database" ]; then
        echo "  Installing gdown for Google Drive download..."
        "$PYTHON" -m pip install -q gdown
        "$PYTHON" -c "
import gdown, zipfile, os
out = os.environ.get('NAS_DIR', '/nas/Dataset/nlp') + '/spider'
zip_path = f'{out}/spider_data.zip'
print('  Downloading Spider DB files from Google Drive...')
gdown.download(id='1iRDVHLr4mX2wQKSgA9OEtSZJqypKHrD', output=zip_path, quiet=False)
if os.path.exists(zip_path) and os.path.getsize(zip_path) > 1e6:
    print('  Extracting...')
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out)
    os.remove(zip_path)
    print('  Spider DB files done.')
else:
    print('  WARNING: DB download failed. Exec eval will not be available.')
    if os.path.exists(zip_path): os.remove(zip_path)
" || echo "  WARNING: Spider DB download failed. Download spider.zip manually."
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
import os, json, subprocess, shutil, csv

nas = os.environ.get("NAS_DIR", "/nas/Dataset/nlp")
out = f"{nas}/cogs"
os.makedirs(out, exist_ok=True)

# COGS: clone from official GitHub (HuggingFace path changed)
repo_dir = f"{out}/_repo"
if not os.path.isdir(repo_dir):
    print("  Cloning COGS from GitHub...")
    subprocess.run(["git", "clone", "--quiet", "--depth=1",
                    "https://github.com/najoungkim/COGS.git", repo_dir], check=True)

data_dir = f"{repo_dir}/data"
for fname in os.listdir(data_dir):
    if fname.endswith(".tsv"):
        shutil.copy(f"{data_dir}/{fname}", f"{out}/{fname}")
        with open(f"{data_dir}/{fname}") as f:
            n = sum(1 for _ in f) - 1
        print(f"  Copied {n} rows -> {out}/{fname}")
print("  COGS done.")
EOF
    echo "  COGS done: $COGS_DIR"
else
    echo "  COGS already present, skipping."
fi

# -------------------------------------------------------
# 3. SCAN (systematic compositional generalization)
# HuggingFace 4.x dropped script-based datasets; clone from GitHub instead.
# -------------------------------------------------------
SCAN_DIR="$NAS_DIR/scan"
if [ ! -f "$SCAN_DIR/simple_train.json" ]; then
    echo "--- Downloading SCAN ---"
    mkdir -p "$SCAN_DIR"
    $PYTHON - <<'EOF'
import os, json, subprocess, re

nas = os.environ.get("NAS_DIR", "/nas/Dataset/nlp")
out = f"{nas}/scan"
os.makedirs(out, exist_ok=True)

repo = f"{out}/_repo"
if not os.path.isdir(repo):
    print("  Cloning SCAN from GitHub...")
    subprocess.run(["git", "clone", "--quiet", "--depth=1",
                    "https://github.com/brendenlake/SCAN.git", repo], check=True)

def parse_scan_file(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            m = re.match(r"IN:\s*(.+?)\s+OUT:\s*(.+)", line)
            if m:
                rows.append({"commands": m.group(1), "actions": m.group(2)})
    return rows

splits = {
    "simple": {
        "train": f"{repo}/simple_split/tasks_train_simple.txt",
        "test":  f"{repo}/simple_split/tasks_test_simple.txt",
    },
    "addprim_jump": {
        "train": f"{repo}/add_prim_split/tasks_train_addprim_jump.txt",
        "test":  f"{repo}/add_prim_split/tasks_test_addprim_jump.txt",
    },
    "length": {
        "train": f"{repo}/length_split/tasks_train_length.txt",
        "test":  f"{repo}/length_split/tasks_test_length.txt",
    },
}
for split_type, files in splits.items():
    for split, path in files.items():
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found, skipping")
            continue
        rows = parse_scan_file(path)
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
# HuggingFace 4.x dropped script-based datasets; use google-research GitHub.
# -------------------------------------------------------
CFQ_DIR="$NAS_DIR/cfq"
if [ ! -f "$CFQ_DIR/mcd1_train.json" ]; then
    echo "--- Downloading CFQ ---"
    mkdir -p "$CFQ_DIR"
    $PYTHON - <<'EOF'
import os, json, subprocess

nas = os.environ.get("NAS_DIR", "/nas/Dataset/nlp")
out = f"{nas}/cfq"
os.makedirs(out, exist_ok=True)

# CFQ is available as a TFRecords file from Google; use the pre-split HF parquet
# Try HF datasets with parquet (doesn't need loading script)
try:
    from datasets import load_dataset
    print("  Trying CFQ from HuggingFace (google-research-datasets/cfq)...")
    for split_type in ["mcd1", "mcd2", "mcd3"]:
        try:
            ds = load_dataset("google-research-datasets/cfq", split_type)
            for split in ds:
                rows = [dict(r) for r in ds[split]]
                fname = f"{split_type}_{split}.json"
                with open(f"{out}/{fname}", "w") as f:
                    json.dump(rows, f, indent=2)
                print(f"  Saved {len(rows)} rows -> {out}/{fname}")
        except Exception as e:
            print(f"  {split_type}: {e}")
except Exception as e:
    print(f"  HF failed: {e}")
    print("  CFQ requires manual download from: https://storage.googleapis.com/cfq_dataset/cfq1.1.tar.gz")

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

nas = os.environ.get("NAS_DIR", "/nas/Dataset/nlp")
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
