#!/usr/bin/env bash
# =============================================================================
# Noor — train our OWN streaming Quran phoneme model (zipformer2-CTC, icefall)
# =============================================================================
# One-paste driver for a RunPod pod. Safe to re-run: every stage is idempotent
# and resumable (training resumes from the last epoch checkpoint).
#
#   Pod: Community Cloud, 1x RTX 4090, 150 GB volume mounted at /workspace,
#        template "RunPod Pytorch 2.4.0" (torch 2.4.0+cu124, python 3.11).
#        Any pod whose torch differs gets the pinned stack installed.
#
#   Run:  bash <(curl -fsSL https://raw.githubusercontent.com/Nour-benmohamed-Git/noor-data/master/train/run_quran_train.sh)
#   ...disconnected? Paste the same line again: it resumes where it stopped.
#
# What it does, in order (markers in /workspace/quran_train/state):
#   S1 deps      pinned k2/lhotse/icefall stack (+ffmpeg, sherpa-onnx for eval)
#   S2 labels    canonical Hafs phoneme labels + 250-unit vocab (noor-data CDN)
#   S3 tools     write the data-prep / tokenizer / datamodule / eval sources
#   S4 data      everyayah -> 16k wavs + lhotse cuts (labels matched by rasm
#                key; ayah-1 gets basmala prepended; dedup + per-reciter cap)
#   S5 features  speed-perturb x3 + 80-dim fbank (train/dev) + MUSAN cuts
#   S6 recipe    icefall@pin, stock zipformer recipe + 2 verified patches
#   S7 smoke     2h subset: 1 epoch -> export -> sherpa decode (mechanics gate)
#   S8 train     full run (default 30 epochs, bf16, chunks 8/16/32/-1)
#   S9 export    epoch-avg -> streaming ONNX (160ms) -> int8 + tokens.txt
#   S10 eval     char-level PER on held-out dev, clean + MUSAN-noised
#
# Cost guide (community RTX 4090 @ ~$0.34/h): prep ~1.5h, train ~1h/epoch.
# The script prints elapsed hours + est. cost after every epoch.
# =============================================================================
set -euo pipefail

# ----------------------------- tunables --------------------------------------
export BASE="${BASE:-/workspace/quran_train}"
export TARGET_TRAIN_HOURS="${TARGET_TRAIN_HOURS:-300}"
export DEV_HOURS="${DEV_HOURS:-2}"
export PER_RECITER_CAP_HOURS="${PER_RECITER_CAP_HOURS:-10}"
export NUM_EPOCHS="${NUM_EPOCHS:-30}"
export MAX_DURATION="${MAX_DURATION:-500}"     # batch seconds (24 GB, bf16); OOM? resume with MAX_DURATION=400
export CHUNK_SIZES="${CHUNK_SIZES:-8,16,32,-1}" # 8 = 160ms profile (the app's)
export EXPORT_CHUNK="${EXPORT_CHUNK:-8}"
export EXPORT_LEFT="${EXPORT_LEFT:-256}"
export EXPORT_AVG="${EXPORT_AVG:-3}"
# Use every GPU on the pod (DDP): a 4x 4090 pod costs the same total dollars
# as 1x for the same epochs, but finishes ~4x sooner.
export WORLD_SIZE="${WORLD_SIZE:-auto}"

ICEFALL_COMMIT="3f848bb6d0acc970c9b294a30ca0a04a7c9c78d1"   # master 2026-07-16
K2_PIN="k2==1.24.4.dev20250715+cuda12.4.torch2.4.0"
TORCH_PIN="2.4.0+cu124"
LABELS_URL="https://cdn.jsdelivr.net/gh/Nour-benmohamed-Git/noor-data@master/train/quran_phoneme_train_labels.v1.json"
MUSAN_URL="https://www.openslr.org/resources/17/musan.tar.gz"

STATE="$BASE/state"; DATA="$BASE/data"; WAVS="$BASE/wavs"; FEATS="$BASE/feats"
RECIPE="$BASE/recipe"; EXP="$BASE/exp"; OUT="$BASE/output"; TOOLS="$BASE/tools"
mkdir -p "$STATE" "$DATA" "$WAVS" "$FEATS" "$RECIPE" "$EXP" "$OUT" "$TOOLS"
cd "$BASE"

log()  { printf '\n\033[1;36m[quran-train]\033[0m %s\n' "$*"; }
fail() { printf '\n\033[1;31m[quran-train] FATAL:\033[0m %s\n' "$*"; exit 1; }
mark() { touch "$STATE/$1.done"; }
done_p() { [ -f "$STATE/$1.done" ]; }

# ----------------------------- S0: preflight ---------------------------------
log "S0 preflight"
command -v nvidia-smi >/dev/null || fail "no nvidia-smi — this is not a GPU pod"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l); [ "$NGPU" -ge 1 ] || NGPU=1
if [ "$WORLD_SIZE" = "auto" ]; then WORLD_SIZE=$NGPU; fi
log "GPUs: $NGPU (training with --world-size $WORLD_SIZE)"
PY=python3
$PY - <<'PYEOF' || fail "python check failed"
import sys
v = sys.version_info
assert (3, 10) <= (v.major, v.minor) <= (3, 11), f"need python 3.10/3.11, got {v}"
print("python", sys.version.split()[0])
PYEOF
AVAIL_GB=$(df -BG --output=avail /workspace | tail -1 | tr -dc '0-9')
if [ ! -f "$STATE/s4_data.done" ]; then
  [ "$AVAIL_GB" -ge 100 ] || fail "need >=100 GB free on /workspace for a fresh run, have ${AVAIL_GB}G"
else
  [ "$AVAIL_GB" -ge 15 ] || fail "volume nearly full (${AVAIL_GB}G free) — delete old epoch checkpoints in $EXP"
fi
log "disk OK (${AVAIL_GB}G free)"

# ----------------------------- S1: dependencies ------------------------------
# S1 runs EVERY paste: pip/apt payloads live on the ephemeral container disk
# (wiped on pod stop/restart) while markers live on the volume, so presence is
# probed at runtime and only missing pieces are (re)installed.
log "S1 dependencies (probe + install missing)"
export DEBIAN_FRONTEND=noninteractive
command -v ffmpeg >/dev/null 2>&1 || {
  apt-get update -qq && apt-get install -y -qq ffmpeg git > /dev/null; }
command -v git >/dev/null 2>&1 || apt-get install -y -qq git > /dev/null

TORCH_VER=$($PY -c "import torch; print(torch.__version__)" 2>/dev/null || echo none)
if [ "$TORCH_VER" != "$TORCH_PIN" ]; then
  log "pod torch is '$TORCH_VER' (want $TORCH_PIN) — installing pinned stack"
  $PY -m pip install --no-cache-dir "torch==$TORCH_PIN" "torchaudio==$TORCH_PIN" \
    -f https://download.pytorch.org/whl/torch/ -f https://download.pytorch.org/whl/torchaudio/
else
  $PY -c "import torchaudio" 2>/dev/null || \
    $PY -m pip install --no-cache-dir "torchaudio==$TORCH_PIN" -f https://download.pytorch.org/whl/torchaudio/
fi
$PY -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"

$PY -c "import k2" 2>/dev/null || \
  $PY -m pip install --no-cache-dir "$K2_PIN" -f https://k2-fsa.github.io/k2/cuda.html
$PY -c "import k2; print('k2', k2.__version__)"

$PY -c "import lhotse, huggingface_hub, pyarrow, soundfile, sherpa_onnx" 2>/dev/null || \
  $PY -m pip install --no-cache-dir lhotse huggingface_hub pyarrow soundfile \
      sherpa-onnx kaldialign sentencepiece tensorboard onnx onnxruntime

if [ ! -d "$BASE/icefall" ]; then
  git clone https://github.com/k2-fsa/icefall.git "$BASE/icefall"
fi
if [ "$(git -C "$BASE/icefall" rev-parse HEAD)" != "$ICEFALL_COMMIT" ]; then
  git -C "$BASE/icefall" fetch --depth 1 origin "$ICEFALL_COMMIT" 2>/dev/null || \
    git -C "$BASE/icefall" fetch origin "$ICEFALL_COMMIT"
  git -C "$BASE/icefall" checkout -q "$ICEFALL_COMMIT"
fi
export PYTHONPATH="$BASE/icefall:${PYTHONPATH:-}"
# icefall.utils has hard imports (pypinyin etc.): install requirements until
# the imports the recipe actually needs succeed — no silent degradation.
$PY -c "import icefall.utils, icefall.checkpoint" 2>/dev/null || {
  $PY -m pip install --no-cache-dir -r "$BASE/icefall/requirements.txt" || true
  $PY -c "import icefall.utils, icefall.checkpoint" || \
    fail "icefall core imports failing even after requirements install"
}

# ----------------------------- S2: labels + vocab ----------------------------
if ! done_p s2_labels; then
  log "S2 fetching phoneme labels + building tokens"
  curl -fsSL "$LABELS_URL" -o "$DATA/labels.json"
  $PY - <<'PYEOF'
import json, os
base = os.environ["BASE"]
d = json.load(open(f"{base}/data/labels.json", encoding="utf-8"))
assert len(d["verses"]) == 6236 and len(d["vocab"]) == 250, "labels pack malformed"
# icefall/sherpa convention: blank at id 0, units at 1..250
with open(f"{base}/data/tokens.txt", "w", encoding="utf-8", newline="\n") as f:
    f.write("<blk> 0\n")
    for i, u in enumerate(d["vocab"]):
        f.write(f"{u} {i + 1}\n")
print("tokens.txt written: 251 entries (blank=0)")
PYEOF
  mark s2_labels
fi

# ----------------------------- S3: tool sources ------------------------------
log "S3 writing tool sources"

cat > "$TOOLS/unit_tokenizer.py" <<'PYEOF'
"""Duck-typed stand-in for sentencepiece over the 250 phoneme units.

Supervision texts are SPACE-SEPARATED phoneme units, so encoding is a split +
vocab lookup. Exposes exactly the SentencePieceProcessor surface icefall's
zipformer train.py uses: load / encode(out_type) / piece_to_id / get_piece_size.
Blank <blk> is id 0 (icefall + sherpa zipformer2-ctc convention; model.py's
F.ctc_loss uses the default blank=0).
"""


class SentencePieceProcessor:
    def __init__(self):
        self._tok2id = {}

    def load(self, path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split(" ")
                if len(parts) >= 2:
                    self._tok2id[parts[0]] = int(parts[1])
        assert self._tok2id.get("<blk>") == 0, "tokens.txt must have <blk> 0"

    def _one(self, text, out_type):
        units = text.split()
        for u in units:
            if u not in self._tok2id:
                raise KeyError(f"unit not in vocab: {u!r}")
        if out_type is str:
            return units
        return [self._tok2id[u] for u in units]

    def encode(self, text, out_type=int):
        if isinstance(text, str):
            return self._one(text, out_type)
        return [self._one(t, out_type) for t in text]

    def piece_to_id(self, piece):
        # <sos/eos> is only used by the transducer decoder (disabled).
        return self._tok2id.get(piece, 0)

    def get_piece_size(self):
        return len(self._tok2id)
PYEOF

cat > "$TOOLS/prepare_quran_data.py" <<'PYEOF'
"""everyayah -> 16 kHz mono wavs + lhotse cuts with phoneme-unit transcripts.

Reads tarteel-ai/everyayah parquet shards directly (HfFileSystem + pyarrow,
one row group at a time — the datasets library chokes on >2GB audio columns).
Rows are matched to canonical verse labels via a rasm-skeleton text key
(diacritics stripped; hamza forms and the volatile weak letters ا/و/ي deleted)
— verified locally: 805 h of 830 h match; ambiguous keys (same skeleton,
different phonemes) are dropped for safety.

Ayah-1 rows: everyayah recitations start surahs with the basmala while the
text column has only the ayah, so ayah-1 labels get the basmala units
PREPENDED (verified: 2:1 'alif-lam-mim' audio is 14.5 s). 1:1 IS the basmala
(kept plain, <=12 s guard); 9:1 has no basmala (skipped).

Resumable: per-shard progress + accumulator state under --out/state.json.
"""
import argparse
import io
import json
import os
import re
import subprocess
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_STRIP = re.compile("[ً-ٰٟـۖ-ۭ]")


def rasm_key(s):
    s = unicodedata.normalize("NFC", s or "")
    s = _STRIP.sub("", s)
    s = s.replace("ى", "ي").replace("ة", "ه")
    s = re.sub("[اويءأإآؤئٱ]", "", s)
    return re.sub(r"\s+", "", s)


def norm_reciter(v):
    s = "" if v is None else str(v)
    return re.sub(r"[^0-9a-z؀-ۿ]", "", s.lower()) or "unknown"


def ffmpeg_decode(raw, out_path):
    p = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", "pipe:0", "-ar", "16000", "-ac", "1",
         "-sample_fmt", "s16", "-y", out_path],
        input=raw, capture_output=True)
    return p.returncode == 0 and os.path.getsize(out_path) > 1000


def load_state(path):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            st = json.load(f)
        st["seen"] = set(tuple(x) for x in st["seen"])
        return st
    return {"seen": set(), "reciter_sec": {}, "train_sec": 0.0, "dev_sec": 0.0,
            "done_shards": [], "rows": []}


def save_state(path, st):
    d = dict(st)
    d["seen"] = sorted(list(x) for x in st["seen"])
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--wavs", required=True)
    ap.add_argument("--target-hours", type=float, required=True)
    ap.add_argument("--dev-hours", type=float, required=True)
    ap.add_argument("--cap-hours", type=float, required=True)
    args = ap.parse_args()

    from huggingface_hub import HfFileSystem
    import pyarrow.parquet as pq

    labels = json.load(open(args.labels, encoding="utf-8"))
    basmala_u = labels["basmala_units"]

    # rasm key -> (verse key, units); ambiguous skeletons dropped
    key2label = {}
    ambiguous = set()
    for vkey, v in labels["verses"].items():
        for k in {rasm_key_from_stored(kk) for kk in v["k"]}:
            if k in key2label and key2label[k][1] != v["u"]:
                ambiguous.add(k)
            else:
                key2label.setdefault(k, (vkey, v["u"]))
    for k in ambiguous:
        key2label.pop(k, None)
    print(f"[prep] label keys: {len(key2label)} (dropped {len(ambiguous)} ambiguous)")

    os.makedirs(args.wavs, exist_ok=True)
    os.makedirs(args.out, exist_ok=True)
    state_path = os.path.join(args.out, "state.json")
    st = load_state(state_path)

    fs = HfFileSystem()
    all_shards = sorted(fs.glob("datasets/tarteel-ai/everyayah/data/*.parquet"))
    dev_shards = [p for p in all_shards if "validation" in p]
    train_shards = [p for p in all_shards if "train-" in p.rsplit("/", 1)[-1]]
    import random
    random.Random(42).shuffle(train_shards)
    print(f"[prep] shards: train={len(train_shards)} dev={len(dev_shards)}")

    pool = ThreadPoolExecutor(max_workers=16)

    def handle_shard(path, split, budget_key, budget_sec):
        if path in st["done_shards"]:
            return
        if st[budget_key] >= budget_sec:
            return
        try:
            fh = fs.open(path, "rb")
            pf = pq.ParquetFile(fh)
        except Exception as e:
            print(f"[prep] skip shard {path.rsplit('/',1)[-1]} (open failed: {e})")
            return
        batch_rows = []
        for batch in pf.iter_batches(batch_size=32, columns=["audio", "duration", "text", "reciter"]):
            if st[budget_key] >= budget_sec:
                break
            names = batch.schema.names
            audios = batch.column(names.index("audio")).to_pylist()
            durs = batch.column(names.index("duration")).to_pylist()
            texts = batch.column(names.index("text")).to_pylist()
            recs = batch.column(names.index("reciter")).to_pylist()
            jobs = []
            for audio, dur, text, rec in zip(audios, durs, texts, recs):
                if st[budget_key] >= budget_sec:
                    break
                dur = float(dur or 0)
                if dur < 1.0 or dur > 30.0:
                    continue
                k = rasm_key(text)
                hit = key2label.get(k)
                if hit is None:
                    continue
                vkey, units = hit
                s_num, a_num = vkey.split(":")
                if a_num == "1":
                    if vkey == "9:1":
                        continue
                    if vkey == "1:1":
                        if dur > 12.0:
                            continue
                    else:
                        units = basmala_u + " " + units
                rkey = norm_reciter(rec)
                dkey = (rkey, vkey)
                if dkey in st["seen"]:
                    continue
                if split == "train":
                    cap = args.cap_hours * 3600
                    if st["reciter_sec"].get(rkey, 0.0) + dur > cap:
                        continue
                raw = (audio or {}).get("bytes")
                if not raw:
                    continue
                name = f"{split}_{rkey}_{vkey.replace(':', '_')}.wav"
                wav_path = os.path.join(args.wavs, name)
                st["seen"].add(dkey)
                if split == "train":
                    st["reciter_sec"][rkey] = st["reciter_sec"].get(rkey, 0.0) + dur
                st[budget_key] += dur
                jobs.append((raw, wav_path, dict(
                    id=name[:-4], wav=wav_path, dur=dur, units=units,
                    vkey=vkey, reciter=rkey, split=split)))
            futures = [(pool.submit(ffmpeg_decode, raw, wp), row) for raw, wp, row in jobs]
            for fut, row in futures:
                if fut.result():
                    batch_rows.append(row)
                else:
                    st[budget_key] -= row["dur"]
                    print(f"[prep] decode failed: {row['id']}")
        st["rows"].extend(batch_rows)
        st["done_shards"].append(path)
        save_state(state_path, st)
        print(f"[prep] {path.rsplit('/',1)[-1]}: train={st['train_sec']/3600:.1f}h "
              f"dev={st['dev_sec']/3600:.1f}h rows={len(st['rows'])}")

    for p in dev_shards:
        handle_shard(p, "dev", "dev_sec", args.dev_hours * 3600)
    for p in train_shards:
        handle_shard(p, "train", "train_sec", args.target_hours * 3600)
    pool.shutdown(wait=True)

    # Build lhotse manifests
    from lhotse import CutSet, Recording, SupervisionSegment
    from lhotse.cut import MonoCut

    def to_cuts(split):
        cuts = []
        for r in st["rows"]:
            if r["split"] != split or not os.path.exists(r["wav"]):
                continue
            rec = Recording.from_file(r["wav"], recording_id=r["id"])
            sup = SupervisionSegment(
                id=r["id"], recording_id=r["id"], start=0.0,
                duration=rec.duration, channel=0, text=r["units"],
                custom={"vkey": r["vkey"], "reciter": r["reciter"]})
            cuts.append(MonoCut(id=r["id"], start=0.0, duration=rec.duration,
                                channel=0, recording=rec, supervisions=[sup]))
        return CutSet.from_cuts(cuts)

    for split in ("train", "dev"):
        cs = to_cuts(split)
        out = os.path.join(args.out, f"quran_cuts_{split}_raw.jsonl.gz")
        cs.to_file(out)
        print(f"[prep] {split}: {len(cs)} cuts, "
              f"{sum(c.duration for c in cs)/3600:.1f} h -> {out}")


def rasm_key_from_stored(k):
    # Stored keys in labels.json were built with an older normalization;
    # re-normalize through the same rasm_key so both sides always agree.
    return rasm_key(k)


if __name__ == "__main__":
    main()
PYEOF

cat > "$TOOLS/make_features.py" <<'PYEOF'
"""Speed-perturb x3 + 80-dim fbank for train/dev cuts, MUSAN cuts + fbank.
Follows the stock icefall librispeech data pipeline (precomputed features,
lilcom storage) so train.py's num_frames-based filtering works unchanged."""
import argparse
import glob
import os

from lhotse import CutSet, Fbank, FbankConfig, LilcomChunkyWriter, Recording
from lhotse.cut import MonoCut


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--feats", required=True)
    ap.add_argument("--musan-dir", default="")
    ap.add_argument("--num-jobs", type=int, default=os.cpu_count() or 8)
    args = ap.parse_args()

    extractor = Fbank(FbankConfig(num_mel_bins=80))

    for split, perturb in (("train", True), ("dev", False)):
        out_manifest = os.path.join(args.data, f"quran_cuts_{split}.jsonl.gz")
        if os.path.exists(out_manifest):
            print(f"[feats] {split} already done")
            continue
        cuts = CutSet.from_file(os.path.join(args.data, f"quran_cuts_{split}_raw.jsonl.gz"))
        if perturb:
            cuts = cuts + cuts.perturb_speed(0.9) + cuts.perturb_speed(1.1)
        cuts = cuts.compute_and_store_features(
            extractor=extractor,
            storage_path=os.path.join(args.feats, f"quran_{split}"),
            num_jobs=args.num_jobs,
            storage_type=LilcomChunkyWriter,
        )
        cuts.to_file(out_manifest)
        print(f"[feats] {split}: {len(cuts)} cuts with features")

    if args.musan_dir:
        out_manifest = os.path.join(args.data, "musan_cuts.jsonl.gz")
        if os.path.exists(out_manifest):
            print("[feats] musan already done")
            return
        wavs = sorted(
            glob.glob(os.path.join(args.musan_dir, "music", "**", "*.wav"), recursive=True)
            + glob.glob(os.path.join(args.musan_dir, "noise", "**", "*.wav"), recursive=True)
        )
        print(f"[feats] musan wavs: {len(wavs)}")
        cuts = []
        for w in wavs:
            rid = os.path.splitext(os.path.basename(w))[0]
            try:
                rec = Recording.from_file(w, recording_id=rid)
            except Exception:
                continue
            cuts.append(MonoCut(id=rid, start=0.0, duration=rec.duration,
                                channel=0, recording=rec))
        musan = CutSet.from_cuts(cuts).cut_into_windows(duration=10.0)
        musan = musan.compute_and_store_features(
            extractor=extractor,
            storage_path=os.path.join(args.feats, "musan"),
            num_jobs=args.num_jobs,
            storage_type=LilcomChunkyWriter,
        )
        musan.to_file(out_manifest)
        print(f"[feats] musan: {len(musan)} cuts with features")


if __name__ == "__main__":
    main()
PYEOF

cat > "$TOOLS/asr_datamodule.py" <<'PYEOF'
"""Quran data module exposing the exact class/API icefall's zipformer train.py
imports (LibriSpeechAsrDataModule). Adapted from the stock icefall module
(Apache-2.0): same arguments, precomputed-feature path, MUSAN CutMix and
SpecAugment; every *_cuts() accessor maps onto the Quran manifests."""
import argparse
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from lhotse import CutSet, Fbank, FbankConfig, load_manifest_lazy
from lhotse.dataset import (
    CutConcatenate,
    CutMix,
    DynamicBucketingSampler,
    K2SpeechRecognitionDataset,
    PrecomputedFeatures,
    SimpleCutSampler,
    SpecAugment,
)
from lhotse.dataset.input_strategies import AudioSamples, OnTheFlyFeatures
from lhotse.utils import fix_random_seed
from torch.utils.data import DataLoader

from icefall.utils import str2bool


class _SeedWorkers:
    def __init__(self, seed: int):
        self.seed = seed

    def __call__(self, worker_id: int):
        fix_random_seed(self.seed + worker_id)


class LibriSpeechAsrDataModule:
    def __init__(self, args: argparse.Namespace):
        self.args = args

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(title="ASR data related options")
        group.add_argument("--full-libri", type=str2bool, default=True)
        group.add_argument("--mini-libri", type=str2bool, default=False)
        group.add_argument("--manifest-dir", type=Path, default=Path("data"))
        group.add_argument("--max-duration", type=int, default=200.0)
        group.add_argument("--bucketing-sampler", type=str2bool, default=True)
        group.add_argument("--num-buckets", type=int, default=30)
        group.add_argument("--concatenate-cuts", type=str2bool, default=False)
        group.add_argument("--duration-factor", type=float, default=1.0)
        group.add_argument("--gap", type=float, default=1.0)
        group.add_argument("--on-the-fly-feats", type=str2bool, default=False)
        group.add_argument("--shuffle", type=str2bool, default=True)
        group.add_argument("--drop-last", type=str2bool, default=True)
        group.add_argument("--return-cuts", type=str2bool, default=True)
        group.add_argument("--num-workers", type=int, default=8)
        group.add_argument("--enable-spec-aug", type=str2bool, default=True)
        group.add_argument("--spec-aug-time-warp-factor", type=int, default=80)
        group.add_argument("--enable-musan", type=str2bool, default=True)
        group.add_argument("--input-strategy", type=str, default="PrecomputedFeatures")

    def train_dataloaders(
        self,
        cuts_train: CutSet,
        sampler_state_dict: Optional[Dict[str, Any]] = None,
    ) -> DataLoader:
        transforms = []
        if self.args.enable_musan:
            logging.info("Enable MUSAN")
            cuts_musan = load_manifest_lazy(self.args.manifest_dir / "musan_cuts.jsonl.gz")
            transforms.append(CutMix(cuts=cuts_musan, p=0.5, snr=(10, 20), preserve_id=True))
        else:
            logging.info("Disable MUSAN")

        if self.args.concatenate_cuts:
            transforms = [
                CutConcatenate(duration_factor=self.args.duration_factor, gap=self.args.gap)
            ] + transforms

        input_transforms = []
        if self.args.enable_spec_aug:
            logging.info("Enable SpecAugment")
            input_transforms.append(
                SpecAugment(
                    time_warp_factor=self.args.spec_aug_time_warp_factor,
                    num_frame_masks=10,
                    features_mask_size=27,
                    num_feature_masks=2,
                    frames_mask_size=100,
                )
            )
        else:
            logging.info("Disable SpecAugment")

        train = K2SpeechRecognitionDataset(
            input_strategy=eval(self.args.input_strategy)(),
            cut_transforms=transforms,
            input_transforms=input_transforms,
            return_cuts=self.args.return_cuts,
        )
        if self.args.on_the_fly_feats:
            train = K2SpeechRecognitionDataset(
                cut_transforms=transforms,
                input_strategy=OnTheFlyFeatures(Fbank(FbankConfig(num_mel_bins=80))),
                input_transforms=input_transforms,
                return_cuts=self.args.return_cuts,
            )

        if self.args.bucketing_sampler:
            train_sampler = DynamicBucketingSampler(
                cuts_train,
                max_duration=self.args.max_duration,
                shuffle=self.args.shuffle,
                num_buckets=self.args.num_buckets,
                drop_last=self.args.drop_last,
            )
        else:
            train_sampler = SimpleCutSampler(
                cuts_train,
                max_duration=self.args.max_duration,
                shuffle=self.args.shuffle,
            )
        if sampler_state_dict is not None:
            train_sampler.load_state_dict(sampler_state_dict)

        seed = torch.randint(0, 100000, ()).item()
        return DataLoader(
            train,
            sampler=train_sampler,
            batch_size=None,
            num_workers=self.args.num_workers,
            persistent_workers=False,
            worker_init_fn=_SeedWorkers(seed),
        )

    def valid_dataloaders(self, cuts_valid: CutSet) -> DataLoader:
        if self.args.on_the_fly_feats:
            validate = K2SpeechRecognitionDataset(
                input_strategy=OnTheFlyFeatures(Fbank(FbankConfig(num_mel_bins=80))),
                return_cuts=self.args.return_cuts,
            )
        else:
            validate = K2SpeechRecognitionDataset(return_cuts=self.args.return_cuts)
        valid_sampler = DynamicBucketingSampler(
            cuts_valid, max_duration=self.args.max_duration, shuffle=False
        )
        return DataLoader(
            validate, sampler=valid_sampler, batch_size=None, num_workers=2,
            persistent_workers=False,
        )

    def test_dataloaders(self, cuts: CutSet) -> DataLoader:
        test = K2SpeechRecognitionDataset(
            input_strategy=(
                OnTheFlyFeatures(Fbank(FbankConfig(num_mel_bins=80)))
                if self.args.on_the_fly_feats
                else eval(self.args.input_strategy)()
            ),
            return_cuts=self.args.return_cuts,
        )
        sampler = DynamicBucketingSampler(cuts, max_duration=self.args.max_duration, shuffle=False)
        return DataLoader(test, batch_size=None, sampler=sampler, num_workers=self.args.num_workers)

    # --- cut accessors: every librispeech name maps onto the Quran manifests ---
    @lru_cache()
    def _train(self) -> CutSet:
        return load_manifest_lazy(self.args.manifest_dir / "quran_cuts_train.jsonl.gz")

    @lru_cache()
    def _dev(self) -> CutSet:
        return load_manifest_lazy(self.args.manifest_dir / "quran_cuts_dev.jsonl.gz")

    def train_clean_5_cuts(self) -> CutSet:
        return self._train()

    def train_clean_100_cuts(self) -> CutSet:
        return self._train()

    def train_clean_360_cuts(self) -> CutSet:
        return CutSet.from_cuts([])

    def train_other_500_cuts(self) -> CutSet:
        return CutSet.from_cuts([])

    def train_all_shuf_cuts(self) -> CutSet:
        return self._train()

    def dev_clean_2_cuts(self) -> CutSet:
        return self._dev()

    def dev_clean_cuts(self) -> CutSet:
        return self._dev()

    def dev_other_cuts(self) -> CutSet:
        return CutSet.from_cuts([])

    def test_clean_cuts(self) -> CutSet:
        return self._dev()

    def test_other_cuts(self) -> CutSet:
        return CutSet.from_cuts([])
PYEOF

cat > "$TOOLS/eval_per.py" <<'PYEOF'
"""Char-level phoneme error rate of the exported int8 streaming model on the
held-out dev set, clean and MUSAN-noised. Char-level PER is the app-faithful
metric (the matcher aligns the raw phoneme character stream)."""
import argparse
import glob
import json
import os
import random

import numpy as np
import soundfile as sf


def edit_distance(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(min(prev[j - 1] + (ca != cb), prev[j] + 1, curr[-1] + 1))
        prev = curr
    return prev[-1]


def decode_file(rec, wav_path, noise=None, snr_db=None):
    import sherpa_onnx  # noqa: F401  (import kept local for clarity)

    samples, sr = sf.read(wav_path, dtype="float32")
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    if noise is not None:
        n = noise
        while len(n) < len(samples):
            n = np.concatenate([n, noise])
        n = n[: len(samples)]
        sp = np.sqrt((samples**2).mean() + 1e-9)
        npow = np.sqrt((n**2).mean() + 1e-9)
        n = n * (sp / npow) * (10 ** (-snr_db / 20))
        samples = np.clip(samples + n, -1.0, 1.0)
    stream = rec.create_stream()
    step = 1600
    for off in range(0, len(samples), step):
        stream.accept_waveform(sr, samples[off : off + step])
        while rec.is_ready(stream):
            rec.decode_stream(stream)
    tail = np.zeros(int(sr * 0.6), dtype="float32")
    stream.accept_waveform(sr, tail)
    stream.input_finished()
    while rec.is_ready(stream):
        rec.decode_stream(stream)
    return rec.get_result(stream).replace(" ", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--dev-manifest", required=True)
    ap.add_argument("--musan-dir", default="")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--report", required=True)
    args = ap.parse_args()

    import sherpa_onnx
    from lhotse import CutSet

    rec = sherpa_onnx.OnlineRecognizer.from_zipformer2_ctc(
        tokens=args.tokens, model=args.model, num_threads=4
    )
    cuts = list(CutSet.from_file(args.dev_manifest))
    random.Random(7).shuffle(cuts)
    cuts = cuts[: args.limit]

    noise = None
    if args.musan_dir:
        noises = sorted(glob.glob(os.path.join(args.musan_dir, "noise", "**", "*.wav"), recursive=True))
        if noises:
            noise, _ = sf.read(noises[len(noises) // 2], dtype="float32")
            if noise.ndim > 1:
                noise = noise.mean(axis=1)

    results = {}
    for label, kwargs in (("clean", {}), ("noisy_snr10", {"noise": noise, "snr_db": 10})):
        if label != "clean" and noise is None:
            continue
        errs = tot = 0
        for c in cuts:
            ref = c.supervisions[0].text.replace(" ", "")
            wav = c.recording.sources[0].source
            hyp = decode_file(rec, wav, **kwargs)
            errs += edit_distance(ref, hyp)
            tot += len(ref)
        per = 100.0 * errs / max(1, tot)
        results[label] = round(per, 2)
        print(f"[eval] {label}: char-PER {per:.2f}%  ({len(cuts)} clips)")

    with open(args.report, "w", encoding="utf-8") as f:
        json.dump({"char_per": results, "clips": len(cuts)}, f, indent=2)
    print(f"[eval] report -> {args.report}")


if __name__ == "__main__":
    main()
PYEOF

cat > "$TOOLS/check_prep.py" <<'PYEOF'
"""Gate: refuse to continue when data prep matched too little audio
(transient HF failures leave shards unprocessed; a re-paste retries them)."""
import json
import os

base = os.environ["BASE"]
st = json.load(open(f"{base}/data/state.json", encoding="utf-8"))
target = float(os.environ["TARGET_TRAIN_HOURS"]) * 3600
got = st["train_sec"]
print(f"[data] matched train audio: {got / 3600:.1f}h (target {target / 3600:.0f}h)")
assert got >= 0.5 * target, (
    f"only {got / 3600:.1f}h matched (<50% of target) — likely transient HF "
    "errors; re-paste the command to retry the remaining shards")
PYEOF

# ----------------------------- S4: data prep ---------------------------------
if ! done_p s4_data; then
  log "S4 downloading + matching everyayah (target ${TARGET_TRAIN_HOURS}h train, ${DEV_HOURS}h dev)"
  $PY "$TOOLS/prepare_quran_data.py" \
    --labels "$DATA/labels.json" --out "$DATA" --wavs "$WAVS" \
    --target-hours "$TARGET_TRAIN_HOURS" --dev-hours "$DEV_HOURS" \
    --cap-hours "$PER_RECITER_CAP_HOURS"
  $PY "$TOOLS/check_prep.py"
  mark s4_data
fi

# ----------------------------- S5: features + musan --------------------------
if ! done_p s5_feats; then
  log "S5 MUSAN download + fbank features (speed-perturb x3)"
  if ! done_p musan_extracted; then
    rm -rf "$BASE/musan"; mkdir -p "$BASE/musan"
    curl -fL --retry 3 -o "$BASE/musan/musan.tar.gz" "$MUSAN_URL"
    tar -xzf "$BASE/musan/musan.tar.gz" -C "$BASE/musan"
    rm -f "$BASE/musan/musan.tar.gz"
    mark musan_extracted
  fi
  $PY "$TOOLS/make_features.py" --data "$DATA" --feats "$FEATS" \
      --musan-dir "$BASE/musan/musan"
  mark s5_feats
fi

# ----------------------------- S6: recipe assembly ---------------------------
if ! done_p s6_recipe; then
  log "S6 assembling recipe (stock zipformer @ ${ICEFALL_COMMIT:0:10} + 2 patches)"
  SRC="$BASE/icefall/egs/librispeech/ASR/zipformer"
  # copy EVERY module in the recipe dir (train.py imports decoder/joiner/
  # attention_decoder/encoder_interface unconditionally); -L dereferences the
  # in-repo symlinks. Our datamodule/tokenizer overwrite the stock ones after.
  cp -fL "$SRC"/*.py "$RECIPE/"
  for f in train.py model.py zipformer.py export-onnx-streaming-ctc.py \
           decoder.py joiner.py encoder_interface.py; do
    [ -f "$RECIPE/$f" ] || fail "recipe file missing after copy: $f"
  done
  cp -f "$TOOLS/unit_tokenizer.py" "$RECIPE/"
  cp -f "$TOOLS/asr_datamodule.py" "$RECIPE/"
  $PY - <<'PYEOF'
import os
recipe = os.path.join(os.environ["BASE"], "recipe", "train.py")
src = open(recipe, encoding="utf-8").read()
patches = [
    # our unit tokenizer instead of sentencepiece (same duck-typed surface)
    ("import sentencepiece as spm", "import unit_tokenizer as spm"),
    # Quran ayahs run long; keep clips up to 30 s (prep already caps at 30)
    ("if c.duration < 1.0 or c.duration > 20.0:",
     "if c.duration < 1.0 or c.duration > 30.0:"),
]
for old, new in patches:
    assert src.count(old) == 1, f"patch anchor not found exactly once: {old!r}"
    src = src.replace(old, new)
open(recipe, "w", encoding="utf-8", newline="\n").write(src)
print("[recipe] train.py patched (tokenizer + 30s filter)")
PYEOF
  mark s6_recipe
fi

train_common_args() {
  echo --world-size "$WORLD_SIZE" \
       --use-transducer 0 --use-ctc 1 \
       --causal 1 --chunk-size "$CHUNK_SIZES" \
       --use-bf16 1 \
       --bpe-model "$DATA/tokens.txt" \
       --manifest-dir "$DATA" \
       --max-duration "$MAX_DURATION" \
       --enable-musan 1 \
       --full-libri 1
}

# ----------------------------- S7: smoke gate --------------------------------
if ! done_p s7_smoke; then
  log "S7 smoke run (2h subset, 1 epoch, then export + sherpa decode)"
  SMOKE="$BASE/smoke"; rm -rf "$SMOKE"; mkdir -p "$SMOKE/data" "$SMOKE/exp"
  $PY - <<'PYEOF'
import os
from lhotse import CutSet
base = os.environ["BASE"]
cuts = CutSet.from_file(f"{base}/data/quran_cuts_train.jsonl.gz")
sub, total = [], 0.0
for c in cuts:
    sub.append(c)
    total += c.duration
    if total >= 2 * 3600:
        break
CutSet.from_cuts(sub).to_file(f"{base}/smoke/data/quran_cuts_train.jsonl.gz")
dev = CutSet.from_file(f"{base}/data/quran_cuts_dev.jsonl.gz")
sdev = [c for i, c in enumerate(dev) if i < 60]
CutSet.from_cuts(sdev).to_file(f"{base}/smoke/data/quran_cuts_dev.jsonl.gz")
print(f"[smoke] subset: {total/3600:.2f}h train, {len(sdev)} dev cuts")
PYEOF
  cp -f "$DATA/musan_cuts.jsonl.gz" "$SMOKE/data/" 2>/dev/null || true
  ( cd "$RECIPE" && $PY train.py $(train_common_args) \
      --manifest-dir "$SMOKE/data" \
      --exp-dir "$SMOKE/exp" --num-epochs 1 --start-epoch 1 \
    2>&1 | tee "$SMOKE/train.log" )
  grep -q "Epoch 1" "$SMOKE/train.log" || fail "smoke train produced no epoch log"
  [ -f "$SMOKE/exp/epoch-1.pt" ] || fail "smoke train produced no checkpoint"

  ( cd "$RECIPE" && $PY export-onnx-streaming-ctc.py \
      --exp-dir "$SMOKE/exp" --tokens "$DATA/tokens.txt" \
      --epoch 1 --avg 1 --use-averaged-model 0 \
      --use-transducer 0 --use-ctc 1 --causal 1 \
      --chunk-size "$EXPORT_CHUNK" --left-context-frames "$EXPORT_LEFT" \
    2>&1 | tee "$SMOKE/export.log" )
  SMOKE_ONNX=$(ls -t "$SMOKE"/exp/*.int8.onnx 2>/dev/null | head -1 || true)
  [ -n "$SMOKE_ONNX" ] || SMOKE_ONNX=$(ls -t "$SMOKE"/exp/*.onnx 2>/dev/null | head -1 || true)
  [ -n "$SMOKE_ONNX" ] || fail "smoke export produced no onnx"

  $PY - "$SMOKE_ONNX" <<'PYEOF'
import os, sys
import numpy as np
import sherpa_onnx
import soundfile as sf
from lhotse import CutSet
base = os.environ["BASE"]
model = sys.argv[1]
rec = sherpa_onnx.OnlineRecognizer.from_zipformer2_ctc(
    tokens=f"{base}/data/tokens.txt", model=model, num_threads=2)
dev = list(CutSet.from_file(f"{base}/smoke/data/quran_cuts_dev.jsonl.gz"))[:2]
for c in dev:
    wav = c.recording.sources[0].source
    samples, sr = sf.read(wav, dtype="float32")
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    s = rec.create_stream()
    s.accept_waveform(sr, samples)
    s.input_finished()
    while rec.is_ready(s):
        rec.decode_stream(s)
    print(f"[smoke] decoded {c.id}: {rec.get_result(s)[:60]!r}")
print("[smoke] MECHANICS OK: train -> export -> sherpa decode all work")
PYEOF
  rm -rf "$SMOKE"
  mark s7_smoke
  log "S7 smoke PASSED (1-epoch output is expected to be near-garbage; the full run fixes that)"
fi

# ----------------------------- S8: full training -----------------------------
if ! done_p s8_train; then
  START=1
  LAST=$(ls "$EXP"/epoch-*.pt 2>/dev/null | sed 's/.*epoch-\([0-9]*\)\.pt/\1/' | sort -n | tail -1 || true)
  if [ -n "${LAST:-}" ]; then START=$((LAST + 1)); fi
  if [ "$START" -gt "$NUM_EPOCHS" ]; then
    log "S8 training already complete (epoch $LAST)"
  else
    log "S8 training epochs $START..$NUM_EPOCHS (resume-safe; re-paste the command to resume)"
    T0=$(date +%s)
    ( cd "$RECIPE" && $PY train.py $(train_common_args) \
        --exp-dir "$EXP" --num-epochs "$NUM_EPOCHS" --start-epoch "$START" \
      2>&1 | tee -a "$BASE/train.log" ) &
    TRAIN_PID=$!
    (
      set +e   # the monitor must never abort anything
      while kill -0 $TRAIN_PID 2>/dev/null; do
        sleep 600
        NOW=$(date +%s)
        H=$(awk -v a="$NOW" -v b="$T0" 'BEGIN{printf "%.1f", (a-b)/3600}')
        C=$(awk -v h="$H" -v g="$WORLD_SIZE" 'BEGIN{printf "%.2f", h*0.34*g}')
        E=$(ls "$EXP"/epoch-*.pt 2>/dev/null | wc -l || true)
        echo "[quran-train] elapsed ${H}h (~\$${C} at \$0.34/h) — epoch checkpoints: ${E:-0}/$NUM_EPOCHS"
        # keep the volume alive: retain only the last 6 epoch checkpoints
        # (resume needs the newest; the final export avg window needs 4)
        ls "$EXP"/epoch-*.pt 2>/dev/null | sed 's/.*epoch-\([0-9]*\)\.pt/\1/' | sort -n | head -n -6 | \
          while read -r OLD; do rm -f "$EXP/epoch-$OLD.pt"; done
      done
    ) &
    MON_PID=$!
    TRAIN_RC=0
    wait $TRAIN_PID || TRAIN_RC=$?
    kill $MON_PID 2>/dev/null || true
    [ "$TRAIN_RC" -eq 0 ] || fail "training exited with $TRAIN_RC — re-paste the command to resume (if CUDA OOM: resume with MAX_DURATION=400)"
  fi
  LASTCK=$(ls "$EXP"/epoch-*.pt 2>/dev/null | sed 's/.*epoch-\([0-9]*\)\.pt/\1/' | sort -n | tail -1 || true)
  [ -n "$LASTCK" ] && [ "$LASTCK" -ge "$NUM_EPOCHS" ] && mark s8_train || \
    fail "training finished but newest checkpoint is epoch-${LASTCK:-none} (< $NUM_EPOCHS)"
fi

# ----------------------------- S9: export ------------------------------------
if ! done_p s9_export; then
  log "S9 exporting streaming ONNX (chunk $EXPORT_CHUNK = 160ms, left $EXPORT_LEFT, avg $EXPORT_AVG)"
  EXPORT_EPOCH=$(ls "$EXP"/epoch-*.pt 2>/dev/null | sed 's/.*epoch-\([0-9]*\)\.pt/\1/' | sort -n | tail -1 || true)
  [ -n "$EXPORT_EPOCH" ] || fail "no epoch checkpoints found in $EXP"
  log "exporting from epoch $EXPORT_EPOCH (avg $EXPORT_AVG)"
  ( cd "$RECIPE" && $PY export-onnx-streaming-ctc.py \
      --exp-dir "$EXP" --tokens "$DATA/tokens.txt" \
      --epoch "$EXPORT_EPOCH" --avg "$EXPORT_AVG" \
      --use-transducer 0 --use-ctc 1 --causal 1 \
      --chunk-size "$EXPORT_CHUNK" --left-context-frames "$EXPORT_LEFT" \
    2>&1 | tee "$BASE/export.log" )
  INT8=$(ls -t "$EXP"/*.int8.onnx 2>/dev/null | head -1 || true)
  FP32=$(ls -t "$EXP"/*.onnx 2>/dev/null | grep -v int8 | head -1 || true)
  [ -n "$INT8" ] || fail "no int8 onnx produced"
  cp -f "$INT8" "$OUT/model.int8.onnx"
  cp -f "$FP32" "$OUT/model.fp32.onnx" 2>/dev/null || true
  cp -f "$DATA/tokens.txt" "$OUT/tokens.txt"
  ( cd "$OUT" && sha1sum model.int8.onnx tokens.txt | tee SHA1SUMS )
  ls -la "$OUT"
  mark s9_export
fi

# ----------------------------- S10: eval -------------------------------------
if ! done_p s10_eval; then
  log "S10 char-PER eval on held-out dev (clean + noisy)"
  $PY "$TOOLS/eval_per.py" \
    --model "$OUT/model.int8.onnx" --tokens "$OUT/tokens.txt" \
    --dev-manifest "$DATA/quran_cuts_dev.jsonl.gz" \
    --musan-dir "$BASE/musan/musan" \
    --limit 300 --report "$OUT/eval_report.json"
  mark s10_eval
fi

log "ALL DONE."
echo "=============================================================="
echo " Artifacts in $OUT :"
echo "   model.int8.onnx  tokens.txt  eval_report.json  SHA1SUMS"
echo " Eval:"
cat "$OUT/eval_report.json" || true
echo
echo " Download them (run on YOUR laptop):"
echo "   runpodctl receive ...   or use the pod's Jupyter file browser"
echo "   (files live at $OUT)"
echo " Then STOP THE POD so it stops billing."
echo "=============================================================="
