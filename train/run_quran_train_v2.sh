#!/usr/bin/env bash
# =============================================================================
# Noor — retrain the streaming Quran phoneme model, v2 (zipformer2-CTC, icefall)
# =============================================================================
# v2 of scripts/runpod_train_quran_phoneme.sh. Same one-paste, marker-resumable
# shape; what changed and why (audit 2026-08-15, all findings verified):
#
#   DATA   everyayah ~346h clean yield (cap 20h/reciter, reservoir-ish sampling,
#          ayah-1 rows skipped: phantom-basmala label noise) PLUS
#          obadx/muaalem-annotated-v3 ~800h (MIT, GOLD phoneme labels in our
#          exact 250-unit script — verified 27/27 configs, zero OOV) PLUS
#          RetaSy learner clips as an EVAL-ONLY real-user set.
#   EVAL   the old dev leaked reciters+verses from train (0.73% PER was
#          memorization). v2 holds reciters ENTIRELY out of train, evaluates
#          unit-PER on 5 sets + mistake/truncation probes, and gates shipping
#          on beating the incumbent model.
#   AUG    ports the missing pieces: RIR reverb view (30% of train), gain
#          jitter +/-10dB (the app feeds unnormalized mic audio), CutMix
#          snr widened to (5,25), RIRS isotropic noises in the pool.
#   PLUMB  manifest shuffle + sampler buffer_size, checkpoint-*.pt retention,
#          constant global batch across pod shapes, exact version pins,
#          sha256-pinned labels, deterministic export names, staged wav
#          deletion, smoke test at the real batch size.
#
# Pod: RunPod Secure Cloud. Day 1 (prep): 1x A100 80GB. Day 2 (train): 4x A100
#      80GB SXM. 350 GB network volume mounted at /workspace (disk peaks at
#      ~280G resident). Template "RunPod Pytorch 2.4.0" (torch 2.4.0+cu124,
#      python 3.11).
#
# Stages (markers in $BASE/state):
#   S0  preflight   S1 deps       S2 labels(+sha256)  S3 tools
#   S4a everyayah   S4b muaalem   S4c retasy(eval)    S4d prep gate
#   S5a EA feats    S5b MU feats  S5c musan+rir+merge
#   S6  recipe      S7 smoke(real batch) S7b ddp sanity
#   S8  train(10ep) S9 export(chunk 8+16, fp32+int8)
#   S10 eval matrix S11 ship gates -> $OUT
#
# Before S10: upload the SHIPPED model for the head-to-head:
#   $BASE/incumbent/model.int8.onnx  (from assets/models/quran-phoneme-160/)
#   $BASE/incumbent/tokens.txt
# =============================================================================
set -euo pipefail

# ----------------------------- tunables --------------------------------------
export BASE="${BASE:-/workspace/quran_train_v2}"

# everyayah's REAL yield under the v2 filters (ayah-1 skip, 1.2-26.5s window,
# 20h/reciter cap, 6 hash-held-out reciters, accept_p) measured 2026-08-17 on a
# full 132-shard pass: 346h. The 700h first guess assumed v1's pre-filter count.
export EA_TRAIN_HOURS="${EA_TRAIN_HOURS:-340}"
export EA_CAP_HOURS="${EA_CAP_HOURS:-20}"
export EA_INDEV_HOURS="${EA_INDEV_HOURS:-1.5}"
export EA_ACCEPT_P="${EA_ACCEPT_P:-0.9}"        # per-row acceptance: spreads picks across each reciter's mushaf
export EA_DEV_MOD="${EA_DEV_MOD:-12}"           # reciters with sha1(name)%MOD==0 held ENTIRELY out of train
export EA_DEV_RECITERS="${EA_DEV_RECITERS:-}"   # optional comma list overriding the hash rule

export MU_TRAIN_HOURS="${MU_TRAIN_HOURS:-800}"
export MU_CAP_HOURS="${MU_CAP_HOURS:-45}"       # per moshaf config
export MU_DEV_XIDS="${MU_DEV_XIDS:-13,19}"      # moshaf_<X>.* held entirely out of train
export MU_MATCH_RATIO_MIN="${MU_MATCH_RATIO_MIN:-0.99}"

export NUM_EPOCHS="${NUM_EPOCHS:-10}"
export GLOBAL_BATCH_SEC="${GLOBAL_BATCH_SEC:-3000}"  # per-GPU max_duration = this / WORLD_SIZE
export TRAIN_GPUS="${TRAIN_GPUS:-4}"                 # intended S8 pod shape; smoke tests THIS batch size
export EXPECTED_GPUS="${EXPECTED_GPUS:-}"            # set on the training pod to assert the pod shape
export CHUNK_SIZES="${CHUNK_SIZES:-8,16,32,-1}"
export EXPORT_CHUNKS="${EXPORT_CHUNKS:-8 16}"        # 160ms and 320ms profiles, both exported
export EXPORT_LEFT="${EXPORT_LEFT:-256}"
export EXPORT_AVG="${EXPORT_AVG:-3}"
export RIR_FRACTION="${RIR_FRACTION:-0.30}"
export GPU_RATE="${GPU_RATE:-1.59}"                  # $/GPU-hour, for the cost monitor
export EVAL_LIMIT="${EVAL_LIMIT:-300}"               # clips per eval set (ea_unseen uses 400)

ICEFALL_COMMIT="3f848bb6d0acc970c9b294a30ca0a04a7c9c78d1"   # master 2026-07-16 (same as v1 run)
K2_PIN="k2==1.24.4.dev20250715+cuda12.4.torch2.4.0"
TORCH_PIN="2.4.0+cu124"
LHOTSE_PIN="lhotse==1.33.0"
QT_PIN="quran-transcript==0.5.2"
LABELS_URL="https://cdn.jsdelivr.net/gh/Nour-benmohamed-Git/noor-data@master/train/quran_phoneme_train_labels.v1.json"
LABELS_SHA256="2319b26699d85a1b9d9bfe4d6e84814c25c391d1d7d066f9c0f29c7997a025a7"
TOKENS_SHA256="bb45c7c6cad391c43ea5cf3b736a8986e974d088032430e81299ae8abf2a5ba8"  # == shipped tokens.tokens
INCUMBENT_SHA1="a75a025de50b35e525384a3a0113a4a859822c80"    # shipped model.int8.onnx
MUSAN_URL="https://www.openslr.org/resources/17/musan.tar.gz"
RIRS_URL="https://www.openslr.org/resources/28/rirs_noises.zip"

export STATE="$BASE/state" DATA="$BASE/data" WAVS="$BASE/wavs" FEATS="$BASE/feats"
export RECIPE="$BASE/recipe" EXP="$BASE/exp" OUT="$BASE/output" TOOLS="$BASE/tools"
export EVALSETS="$BASE/evalsets"
export LABELS_SHA256 TOKENS_SHA256
mkdir -p "$STATE" "$DATA" "$WAVS" "$FEATS" "$RECIPE" "$EXP" "$OUT" "$TOOLS" "$EVALSETS"
cd "$BASE"

log()  { printf '\n\033[1;36m[quran-v2]\033[0m %s\n' "$*"; }
fail() { printf '\n\033[1;31m[quran-v2] FATAL:\033[0m %s\n' "$*"; exit 1; }
mark() { touch "$STATE/$1.done"; }
done_p() { [ -f "$STATE/$1.done" ]; }

# ----------------------------- S0: preflight ---------------------------------
log "S0 preflight"
command -v nvidia-smi >/dev/null || fail "no nvidia-smi — this is not a GPU pod"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
PY=python3
$PY - <<'PYEOF' || fail "python check failed"
import sys
v = sys.version_info
assert (3, 10) <= (v.major, v.minor) <= (3, 11), f"need python 3.10/3.11, got {v}"
print("python", sys.version.split()[0])
PYEOF
# (GPU sizing happens after S1, once the pinned torch stack is guaranteed)

# capacity gate (free-space would block resumes once prep data lands on disk);
# real peak is ~278G resident during S5b (EA feats 79G + MU wavs 92G + MU feats 90G + noise corpora)
AVAIL_GB=$(df -BG --output=avail "$BASE" 2>/dev/null | tail -1 | tr -dc '0-9')
AVAIL_GB=${AVAIL_GB:-0}
TOTAL_GB=$(df -BG --output=size "$BASE" 2>/dev/null | tail -1 | tr -dc '0-9')
TOTAL_GB=${TOTAL_GB:-0}
if [ ! -f "$STATE/s5c_merge.done" ]; then
  [ "$TOTAL_GB" -ge 330 ] || fail "volume is ${TOTAL_GB}G; v2 peaks at ~280G resident — use a 350G volume"
  [ "$AVAIL_GB" -ge 20 ] || fail "only ${AVAIL_GB}G free mid-prep — stale data on the volume? inspect $BASE"
else
  [ "$AVAIL_GB" -ge 25 ] || fail "volume nearly full (${AVAIL_GB}G free) — prune $EXP"
fi
log "disk OK (${AVAIL_GB}G free of ${TOTAL_GB}G)"

# ----------------------------- S1: dependencies ------------------------------
# Runs EVERY paste: pip/apt payloads live on the ephemeral container disk while
# markers live on the volume, so presence is probed and only gaps installed.
log "S1 dependencies (probe + install missing)"
export DEBIAN_FRONTEND=noninteractive
command -v ffmpeg >/dev/null 2>&1 || {
  apt-get update -qq && apt-get install -y -qq ffmpeg git unzip > /dev/null; }
command -v git >/dev/null 2>&1 || apt-get install -y -qq git > /dev/null
command -v unzip >/dev/null 2>&1 || apt-get install -y -qq unzip > /dev/null

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
$PY -c "import k2; print('k2', getattr(k2, '__version__', getattr(k2, '__dev_version__', 'ok')))"

# lhotse pinned (it defines the train-time feature contract); sherpa/onnxruntime
# recorded to $OUT for reproducibility.
$PY -c "import lhotse; assert lhotse.__version__.startswith('1.33')" 2>/dev/null || \
  $PY -m pip install --no-cache-dir "$LHOTSE_PIN"
$PY -c "import huggingface_hub, pyarrow, soundfile, sherpa_onnx, lilcom" 2>/dev/null || \
  $PY -m pip install --no-cache-dir lilcom huggingface_hub pyarrow soundfile \
      sherpa-onnx kaldialign sentencepiece tensorboard onnx onnxruntime
$PY -c "import quran_transcript" 2>/dev/null || \
  $PY -m pip install --no-cache-dir "$QT_PIN"
$PY - <<'PYEOF'
import os
import lhotse, sherpa_onnx, onnxruntime, onnx
vers = {
    "lhotse": lhotse.__version__,
    "sherpa_onnx": sherpa_onnx.__version__,
    "onnxruntime": onnxruntime.__version__,
    "onnx": onnx.__version__,
}
with open(os.path.join(os.environ["OUT"], "pip_versions.txt"), "w") as f:
    for k, v in vers.items():
        f.write(f"{k}=={v}\n")
print("[deps]", vers)
PYEOF

if [ ! -d "$BASE/icefall" ]; then
  git clone https://github.com/k2-fsa/icefall.git "$BASE/icefall"
fi
if [ "$(git -C "$BASE/icefall" rev-parse HEAD)" != "$ICEFALL_COMMIT" ]; then
  git -C "$BASE/icefall" fetch --depth 1 origin "$ICEFALL_COMMIT" 2>/dev/null || \
    git -C "$BASE/icefall" fetch origin "$ICEFALL_COMMIT"
  git -C "$BASE/icefall" checkout -q "$ICEFALL_COMMIT"
fi
export PYTHONPATH="$BASE/icefall:${PYTHONPATH:-}"
$PY -c "import icefall.utils, icefall.checkpoint" 2>/dev/null || {
  $PY -m pip install --no-cache-dir -r "$BASE/icefall/requirements.txt" || true
  $PY -c "import icefall.utils, icefall.checkpoint" || \
    fail "icefall core imports failing even after requirements install"
}

# GPU sizing — after S1 so the pinned torch stack is in place. HARD-FAIL on
# 0/mismatch: a silent world-size-1 fallback would run the 4-GPU job at 4x the
# wall clock and cost, or OOM at max_duration 3000.
WORLD_SIZE=$($PY -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 0)
SMI_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
[ "$WORLD_SIZE" -ge 1 ] || fail "torch reports 0 CUDA devices (nvidia-smi sees $SMI_GPUS)"
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ] && [ "$WORLD_SIZE" != "$SMI_GPUS" ]; then
  fail "torch sees $WORLD_SIZE GPUs but nvidia-smi lists $SMI_GPUS — broken driver/stack"
fi
if [ -n "$EXPECTED_GPUS" ] && [ "$WORLD_SIZE" != "$EXPECTED_GPUS" ]; then
  fail "pod has $WORLD_SIZE visible GPUs but EXPECTED_GPUS=$EXPECTED_GPUS — wrong pod shape"
fi
export WORLD_SIZE
export MAX_DURATION=$(( GLOBAL_BATCH_SEC / WORLD_SIZE ))
log "GPUs: $WORLD_SIZE  (per-GPU max_duration $MAX_DURATION s; global batch $GLOBAL_BATCH_SEC s)"
echo "WORLD_SIZE=$WORLD_SIZE MAX_DURATION=$MAX_DURATION GLOBAL_BATCH_SEC=$GLOBAL_BATCH_SEC NUM_EPOCHS=$NUM_EPOCHS" > "$OUT/run_config.txt"

# ----------------------------- S2: labels + vocab ----------------------------
if ! done_p s2_labels; then
  log "S2 fetching phoneme labels + building tokens (sha256-pinned)"
  curl -fsSL "$LABELS_URL" -o "$DATA/labels.json"
  $PY - <<'PYEOF'
import hashlib, json, os
base = os.environ["BASE"]
raw = open(f"{base}/data/labels.json", "rb").read()
got = hashlib.sha256(raw).hexdigest()
want = os.environ["LABELS_SHA256"]
assert got == want, f"labels.json sha256 mismatch: {got} != {want} (CDN drift — do not train)"
d = json.loads(raw)
assert len(d["verses"]) == 6236 and len(d["vocab"]) == 250, "labels pack malformed"
# icefall/sherpa convention: blank at id 0, units at 1..250
lines = ["<blk> 0\n"] + [f"{u} {i + 1}\n" for i, u in enumerate(d["vocab"])]
data = "".join(lines).encode("utf-8")
tok = hashlib.sha256(data).hexdigest()
want_tok = os.environ["TOKENS_SHA256"]
assert tok == want_tok, f"generated tokens.txt sha256 {tok} != shipped {want_tok} — app contract broken"
with open(f"{base}/data/tokens.txt", "wb") as f:
    f.write(data)
print("tokens.txt written: 251 entries (blank=0), byte-identical to the shipped file")
PYEOF
  mark s2_labels
fi
export LABELS_SHA256 TOKENS_SHA256

# ----------------------------- S3: tool sources ------------------------------
# Rewritten EVERY paste; the recipe copies (after S6) also refresh every paste
# so edits to these sources reach training without stale-marker traps.
log "S3 writing tool sources"

cat > "$TOOLS/unit_tokenizer.py" <<'PYEOF'
"""Duck-typed stand-in for sentencepiece over the 250 phoneme units.

Supervision texts are SPACE-SEPARATED phoneme units, so encoding is a split +
vocab lookup. Exposes exactly the SentencePieceProcessor surface icefall's
zipformer train.py uses. Blank <blk> is id 0. Raises KeyError on any
out-of-vocab unit — that hard failure is a data-integrity guard, keep it.
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
        return self._tok2id.get(piece, 0)

    def get_piece_size(self):
        return len(self._tok2id)
PYEOF

cat > "$TOOLS/prep_common.py" <<'PYEOF'
"""Shared helpers for the v2 data preps: two-tier verse-text matching,
reciter normalization, ffmpeg decode, resumable state."""
import hashlib
import json
import os
import re
import subprocess
import unicodedata

# Diacritics + Quranic annotation signs (U+064B-0670, U+06D6-06ED as in v1)
# plus tatweel AND Arabic Extended-A combining marks U+08D3-08FF (open tanwin
# etc., used by some Uthmani texts e.g. RetaSy); digits are NOT touched.
_STRIP = re.compile("[ً-ٰٟـۖ-ۭ࣓-ࣿ]")


def mild_key(s):
    """Tier-1 normalization: strip diacritics, fold hamza/yeh/teh-marbuta
    variants, KEEP the weak letters. Zero ambiguity on the real verse table."""
    s = unicodedata.normalize("NFC", s or "")
    s = _STRIP.sub("", s)
    for a, b in (("أ", "ا"), ("إ", "ا"), ("آ", "ا"), ("ٱ", "ا"),
                 ("ؤ", "و"), ("ئ", "ي"), ("ى", "ي"), ("ی", "ي"),
                 ("ک", "ك"), ("ة", "ه")):
        s = s.replace(a, b)
    return re.sub(r"\s+", "", s)


def alif_key(s):
    """Tier-1.5: mild + delete alif only. Unifies defective-alif spellings
    (Uthmani العلمين vs imlaa'i العالمين) WITHOUT the waw-deletion collision
    class of the rasm tier (which merged 1:2 into 37:182 and lost Al-Fatiha)."""
    return mild_key(s).replace("ا", "")


def rasm_key(s):
    """Tier-2 fallback: v1's aggressive skeleton (also deletes weak letters).
    Last resort when mild and alif both miss."""
    s = unicodedata.normalize("NFC", s or "")
    s = _STRIP.sub("", s)
    s = s.replace("ی", "ي").replace("ى", "ي").replace("ة", "ه")
    s = re.sub("[اويءأإآؤئٱ]", "", s)
    return re.sub(r"\s+", "", s)


def build_matchers(labels):
    """Returns (mild, alif, rasm) dicts: key -> (vkey, units, group_size).
    group_size = number of verses sharing that exact text (identical units
    after the ayah-1 skip), so dedup can allow that many occurrences."""
    def build(keyfn):
        groups = {}
        for vkey, v in labels["verses"].items():
            for kk in v["k"]:
                k = keyfn(kk)
                groups.setdefault(k, []).append((vkey, v["u"]))
        out, ambiguous = {}, 0
        for k, entries in groups.items():
            units = {u for _, u in entries}
            if len(units) > 1:
                ambiguous += 1
                continue
            vkeys = sorted({vk for vk, _ in entries})
            out[k] = (vkeys[0], entries[0][1], len(vkeys))
        return out, ambiguous

    mild, mild_amb = build(mild_key)
    alif, alif_amb = build(alif_key)
    rasm, rasm_amb = build(rasm_key)
    print(f"[match] mild keys={len(mild)} (amb dropped {mild_amb}); "
          f"alif keys={len(alif)} (amb dropped {alif_amb}); "
          f"rasm keys={len(rasm)} (amb dropped {rasm_amb})")
    return mild, alif, rasm


def match_verse(text, matchers, tier_hits):
    mild, alif, rasm = matchers
    for name, keyfn, table in (("mild", mild_key, mild),
                               ("alif", alif_key, alif),
                               ("rasm", rasm_key, rasm)):
        hit = table.get(keyfn(text))
        if hit is not None:
            tier_hits[name] += 1
            return hit
    tier_hits["miss"] += 1
    return None


def norm_reciter(v):
    s = "" if v is None else str(v)
    return re.sub(r"[^0-9a-z؀-ۿ]", "", s.lower()) or "unknown"


def stable_hash(s):
    return int(hashlib.sha1(s.encode("utf-8")).hexdigest()[:8], 16)


def ffmpeg_decode(raw, out_path):
    p = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", "pipe:0", "-ar", "16000", "-ac", "1",
         "-sample_fmt", "s16", "-y", out_path],
        input=raw, capture_output=True)
    return p.returncode == 0 and os.path.getsize(out_path) > 1000


def save_state(path, st):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(st, f, ensure_ascii=False)
    os.replace(tmp, path)


def write_jsonl(path, rows):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def read_jsonl(path):
    out = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    # a pod kill mid-append can tear the final line; the row's
                    # shard is not marked done, so it re-emits on this paste
                    print(f"[warn] dropping truncated row in {path}")
    return out


def rows_to_cuts(rows, out_manifest):
    """rows: dicts with id/wav/units. Writes a lhotse CutSet manifest."""
    from lhotse import CutSet, Recording, SupervisionSegment
    from lhotse.cut import MonoCut
    cuts = []
    for r in rows:
        if not os.path.exists(r["wav"]):
            continue
        rec = Recording.from_file(r["wav"], recording_id=r["id"])
        sup = SupervisionSegment(
            id=r["id"], recording_id=r["id"], start=0.0,
            duration=rec.duration, channel=0, text=r["units"],
            custom={"vkey": r.get("vkey", ""), "reciter": r.get("reciter", "")})
        cuts.append(MonoCut(id=r["id"], start=0.0, duration=rec.duration,
                            channel=0, recording=rec, supervisions=[sup]))
    cs = CutSet.from_cuts(cuts)
    cs.to_file(out_manifest)
    return len(cuts), sum(c.duration for c in cuts)
PYEOF

cat > "$TOOLS/prepare_everyayah.py" <<'PYEOF'
"""everyayah -> 16 kHz wavs + train/eval rows (v2).

v2 changes vs v1 (audit-driven):
- ALL ayah-1 rows SKIPPED, 1:1 included (phantom basmala/isti'adha prefixes,
  the 39:1/45:2/46:2 wrong-basmala bug; muaalem covers these acoustics with
  correctly-prefixed gold labels).
- dedup allows one occurrence per verse in each identical-text group
  ((reciter, vkey, occurrence) not (reciter, vkey)): recovers the 183
  refrain/muqatta'at verses.
- two-tier text matching (mild first, rasm skeleton fallback) with per-tier
  hit counts printed.
- reciters with sha1(name)%EA_DEV_MOD==0 (or EA_DEV_RECITERS) held ENTIRELY
  out of train -> the EA-unseen eval set. In-reciter dev comes from the
  validation shards, shuffled, per-reciter capped.
- per-row acceptance probability spreads sampling across each reciter's
  mushaf (shards are reciter-major; a prefix cap truncated coverage).
- rows JSONL sidecar; a shard is marked done only when fully consumed.
- verse coverage report.
"""
import argparse
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor

from prep_common import (build_matchers, ffmpeg_decode, match_verse,
                         norm_reciter, save_state, stable_hash, write_jsonl,
                         read_jsonl, rows_to_cuts)

DUR_MIN, DUR_MAX = 1.2, 26.5  # 26.5 * (1/0.9) = 29.4s < the 30.5s train filter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--wavs", required=True)
    ap.add_argument("--evalsets", required=True)
    ap.add_argument("--target-hours", type=float, required=True)
    ap.add_argument("--cap-hours", type=float, required=True)
    ap.add_argument("--indev-hours", type=float, required=True)
    ap.add_argument("--accept-p", type=float, required=True)
    ap.add_argument("--dev-mod", type=int, required=True)
    ap.add_argument("--dev-reciters", default="")
    args = ap.parse_args()

    from huggingface_hub import HfFileSystem
    import pyarrow.parquet as pq

    labels = json.load(open(args.labels, encoding="utf-8"))
    matchers = build_matchers(labels)

    forced_dev = {norm_reciter(x) for x in args.dev_reciters.split(",") if x.strip()}

    def is_dev_reciter(rkey):
        if forced_dev:
            return rkey in forced_dev
        return stable_hash("ea:" + rkey) % args.dev_mod == 0

    os.makedirs(args.wavs, exist_ok=True)
    os.makedirs(args.out, exist_ok=True)
    state_path = os.path.join(args.out, "ea_state.json")
    rows_path = os.path.join(args.out, "ea_rows.jsonl")
    if os.path.exists(state_path):
        st = json.load(open(state_path, encoding="utf-8"))
    else:
        st = {"done_shards": [], "reciter_sec": {}, "train_sec": 0.0,
              "indev_sec": 0.0, "unseen_sec": {}, "occ": {},
              "tier_hits": {"mild": 0, "alif": 0, "rasm": 0, "miss": 0}}
    st["tier_hits"].setdefault("alif", 0)  # states saved before the alif tier existed
    rows = read_jsonl(rows_path)
    rows_f = open(rows_path, "a", encoding="utf-8")

    fs = HfFileSystem()
    all_shards = sorted(fs.glob("datasets/tarteel-ai/everyayah/data/*.parquet"))
    dev_shards = [p for p in all_shards if "validation" in p]
    train_shards = [p for p in all_shards if "train-" in p.rsplit("/", 1)[-1]]
    random.Random(42).shuffle(train_shards)
    random.Random(43).shuffle(dev_shards)
    print(f"[ea] shards: train={len(train_shards)} validation={len(dev_shards)}")

    pool = ThreadPoolExecutor(max_workers=16)
    tier_hits = st["tier_hits"]

    def emit(raw, name, meta):
        wav_path = os.path.join(args.wavs, name)
        return pool.submit(ffmpeg_decode, raw, wav_path), dict(meta, wav=wav_path)

    def handle_shard(path, source):
        """source: 'train' (train + unseen eval) or 'validation' (in-dev)."""
        if path in st["done_shards"]:
            return
        if source == "train" and st["train_sec"] >= args.target_hours * 3600 \
           and sum(st["unseen_sec"].values()) >= 3 * 3600:
            return
        if source == "validation" and st["indev_sec"] >= args.indev_hours * 3600:
            return
        for attempt in range(4):
            try:
                fh = fs.open(path, "rb")
                pf = pq.ParquetFile(fh)
                break
            except Exception as e:
                print(f"[ea] open retry {attempt} {path.rsplit('/', 1)[-1]}: {e}")
                if attempt == 3:
                    return
        snap = json.loads(json.dumps(st))  # restored on mid-shard failure
        futures = []
        try:
            for batch in pf.iter_batches(batch_size=32,
                                         columns=["audio", "duration", "text", "reciter"]):
                names = batch.schema.names
                audios = batch.column(names.index("audio")).to_pylist()
                durs = batch.column(names.index("duration")).to_pylist()
                texts = batch.column(names.index("text")).to_pylist()
                recs = batch.column(names.index("reciter")).to_pylist()
                for audio, dur, text, rec in zip(audios, durs, texts, recs):
                    dur = float(dur or 0)
                    if dur < DUR_MIN or dur > DUR_MAX:
                        continue
                    raw = (audio or {}).get("bytes")
                    if not raw:
                        continue
                    hit = match_verse(text or "", matchers, tier_hits)
                    if hit is None:
                        continue
                    vkey, units, group_size = hit
                    s_num, a_num = vkey.split(":")
                    if a_num == "1":
                        # ALL ayah-1 rows skipped — recordings carry unlabeled
                        # basmala/isti'adha prefixes (even 1:1: isti'adha+basmala
                        # fits any duration guard), and 39:1's text-twins
                        # 45:2/46:2 are mid-surah. Basmala/isti'adha acoustics
                        # come from muaalem rows whose gold labels INCLUDE them.
                        continue
                    rkey = norm_reciter(rec)
                    dev_r = is_dev_reciter(rkey)
                    if source == "validation":
                        if dev_r or st["indev_sec"] >= args.indev_hours * 3600:
                            continue
                        # per-reciter cap keeps the in-dev multi-reciter
                        if st["reciter_sec"].get("indev:" + rkey, 0.0) > 600:
                            continue
                        okey = f"indev|{rkey}|{vkey}"
                        occ = st["occ"].get(okey, 0)
                        if occ >= group_size:
                            continue
                        st["occ"][okey] = occ + 1
                        st["reciter_sec"]["indev:" + rkey] = \
                            st["reciter_sec"].get("indev:" + rkey, 0.0) + dur
                        st["indev_sec"] += dur
                        name = f"eaindev_{rkey}_{vkey.replace(':', '_')}_{occ}.wav"
                        split = "ea_indev"
                    elif dev_r:
                        # held-out reciter -> EA-unseen eval (cap 1.5h each)
                        if st["unseen_sec"].get(rkey, 0.0) > 1.5 * 3600:
                            continue
                        okey = f"unseen|{rkey}|{vkey}"
                        occ = st["occ"].get(okey, 0)
                        if occ >= group_size:
                            continue
                        st["occ"][okey] = occ + 1
                        st["unseen_sec"][rkey] = st["unseen_sec"].get(rkey, 0.0) + dur
                        name = f"eaunseen_{rkey}_{vkey.replace(':', '_')}_{occ}.wav"
                        split = "ea_unseen"
                    else:
                        if st["train_sec"] >= args.target_hours * 3600:
                            continue
                        if st["reciter_sec"].get(rkey, 0.0) + dur > args.cap_hours * 3600:
                            continue
                        # seeded per-row acceptance spreads picks across the mushaf
                        if (stable_hash(f"acc|{rkey}|{vkey}|{dur:.2f}") % 1000) \
                                >= args.accept_p * 1000:
                            continue
                        okey = f"train|{rkey}|{vkey}"
                        occ = st["occ"].get(okey, 0)
                        if occ >= group_size:
                            continue
                        st["occ"][okey] = occ + 1
                        st["reciter_sec"][rkey] = st["reciter_sec"].get(rkey, 0.0) + dur
                        st["train_sec"] += dur
                        name = f"eatrain_{rkey}_{vkey.replace(':', '_')}_{occ}.wav"
                        split = "ea_train"
                    futures.append(emit(raw, name, dict(
                        id=name[:-4], dur=dur, units=units, vkey=vkey,
                        reciter=rkey, split=split)))
        except Exception as e:
            print(f"[ea] shard read failed mid-stream, will retry on next paste: {e}")
            for fut, row in futures:
                fut.result()
            # roll back this shard's counter mutations (shard is NOT marked done)
            for k in ("reciter_sec", "train_sec", "indev_sec", "unseen_sec", "occ"):
                st[k] = snap[k]
            st["tier_hits"].clear()
            st["tier_hits"].update(snap["tier_hits"])
            return
        ok = 0
        for fut, row in futures:
            if fut.result():
                rows.append(row)
                rows_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                ok += 1
        rows_f.flush()
        st["done_shards"].append(path)
        save_state(state_path, st)
        print(f"[ea] {path.rsplit('/', 1)[-1]}: +{ok} rows | train={st['train_sec'] / 3600:.1f}h "
              f"indev={st['indev_sec'] / 3600:.1f}h unseen={sum(st['unseen_sec'].values()) / 3600:.1f}h "
              f"tiers={tier_hits}")

    for p in dev_shards:
        handle_shard(p, "validation")
    for p in train_shards:
        handle_shard(p, "train")
        if (st["train_sec"] >= args.target_hours * 3600
                and len(st["unseen_sec"]) < 2 and not forced_dev):
            # fail NOW, not after streaming every remaining shard: the roster
            # is known, so the operator can pick held-out reciters by name
            roster = sorted(k for k in st["reciter_sec"] if not k.startswith("indev:"))
            pool.shutdown(wait=True)
            save_state(state_path, st)
            raise AssertionError(
                "train target reached but <2 reciters selected by the hash holdout; "
                f"re-paste with EA_DEV_RECITERS=<2-3 of: {roster}> — no re-download needed")
    pool.shutdown(wait=True)
    save_state(state_path, st)

    # a crash mid-shard can leave sidecar rows for a shard that later re-ran;
    # keep the last occurrence of each id
    rows = list({r["id"]: r for r in rows}.values())

    # splits are re-derived from the CURRENT holdout rule, so a re-paste with
    # EA_DEV_RECITERS set recovers a failed holdout WITHOUT re-downloading
    # (wav prefixes may then disagree with splits; deletion is manifest-driven)
    for r in rows:
        if r["split"] in ("ea_train", "ea_unseen"):
            r["split"] = "ea_unseen" if is_dev_reciter(r["reciter"]) else "ea_train"

    train_rows = [r for r in rows if r["split"] == "ea_train"]
    n, sec = rows_to_cuts(train_rows, os.path.join(args.out, "ea_cuts_train_raw.jsonl.gz"))
    print(f"[ea] train cuts: {n} ({sec / 3600:.1f} h)")
    indev_rows = [r for r in rows if r["split"] == "ea_indev"]
    n, sec = rows_to_cuts(indev_rows, os.path.join(args.out, "ea_cuts_indev_raw.jsonl.gz"))
    print(f"[ea] indev cuts: {n} ({sec / 3600:.1f} h)")
    write_jsonl(os.path.join(args.evalsets, "ea_indev.jsonl"), indev_rows)
    unseen_rows = [r for r in rows if r["split"] == "ea_unseen"]
    write_jsonl(os.path.join(args.evalsets, "ea_unseen.jsonl"), unseen_rows)
    dev_names = sorted({r["reciter"] for r in unseen_rows})
    print(f"[ea] unseen eval: {len(unseen_rows)} clips from {len(dev_names)} held-out reciters: {dev_names}")
    assert len(dev_names) >= 2, (
        "fewer than 2 held-out reciters selected by the hash rule — rerun with "
        "EA_DEV_RECITERS=<name1,name2,name3> (see reciter names in ea_state.json)")

    # verse coverage report
    vk = {}
    for r in train_rows:
        vk[r["vkey"]] = vk.get(r["vkey"], 0) + 1
    counts = sorted(vk.values())
    print(f"[ea] verse coverage: {len(vk)}/6236 distinct verses; "
          f"min={counts[0] if counts else 0} median={counts[len(counts) // 2] if counts else 0} "
          f"per covered verse")
    with open(os.path.join(args.out, "ea_coverage.json"), "w", encoding="utf-8") as f:
        json.dump({"distinct_verses": len(vk), "train_hours": st["train_sec"] / 3600,
                   "held_out_reciters": dev_names, "tier_hits": tier_hits}, f)


if __name__ == "__main__":
    main()
PYEOF

cat > "$TOOLS/prepare_muaalem.py" <<'PYEOF'
"""obadx/muaalem-annotated-v3 -> 16 kHz wavs + train/eval rows.

GOLD phoneme labels: the `phonemes` column is the actual realization in the
same Quran Phonetic Script our vocab uses (verified 27/27 audio configs, zero
OOV). Tokenized with quran_transcript.chunck_phonemes; every unit must be in
the 250-unit vocab or the row is counted OOV and the run refuses to continue.

Rows flagged has_istiaatha/has_bismillah are KEPT (their phonemes include the
prefix — real labels, no guessing). has_sadaka rows are skipped (post-Quran
formula). match_ratio < threshold skipped. moshaf_<X>.* configs with X in
--dev-xids are held ENTIRELY out of train -> the MU-unseen eval set.
"""
import argparse
import json
import os
import re
import random
from concurrent.futures import ThreadPoolExecutor

from prep_common import (ffmpeg_decode, save_state, write_jsonl, read_jsonl,
                         rows_to_cuts)

DUR_MIN, DUR_MAX = 1.2, 26.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--wavs", required=True)
    ap.add_argument("--evalsets", required=True)
    ap.add_argument("--target-hours", type=float, required=True)
    ap.add_argument("--cap-hours", type=float, required=True)
    ap.add_argument("--dev-xids", required=True)
    ap.add_argument("--match-ratio-min", type=float, required=True)
    args = ap.parse_args()

    from huggingface_hub import HfFileSystem
    import pyarrow.parquet as pq
    from quran_transcript import chunck_phonemes

    vocab = set()
    with open(args.tokens, encoding="utf-8") as f:
        for line in f:
            u = line.rstrip("\n").rsplit(" ", 1)[0]
            if u != "<blk>":
                vocab.add(u)
    assert len(vocab) == 250

    dev_xids = {x.strip() for x in args.dev_xids.split(",") if x.strip()}

    os.makedirs(args.wavs, exist_ok=True)
    state_path = os.path.join(args.out, "mu_state.json")
    rows_path = os.path.join(args.out, "mu_rows.jsonl")
    if os.path.exists(state_path):
        st = json.load(open(state_path, encoding="utf-8"))
    else:
        st = {"done_files": [], "cfg_sec": {}, "train_sec": 0.0,
              "unseen_sec": 0.0, "oov_rows": 0, "skip": {"ratio": 0, "sadaka": 0, "dur": 0}}
    rows = read_jsonl(rows_path)
    rows_f = open(rows_path, "a", encoding="utf-8")

    fs = HfFileSystem()
    files = sorted(fs.glob("datasets/obadx/muaalem-annotated-v3/**/*.parquet"))
    cfg_files = {}
    for p in files:
        m = re.search(r"/(moshaf_\d+\.\d+)/", p)
        if m:
            cfg_files.setdefault(m.group(1), []).append(p)
    print(f"[mu] audio configs: {len(cfg_files)}")
    assert len(cfg_files) >= 20, "muaalem parquet layout changed — investigate before training"
    have_xids = {c.split("_")[1].split(".")[0] for c in cfg_files}
    assert dev_xids <= have_xids, (
        f"MU_DEV_XIDS {sorted(dev_xids - have_xids)} not present in the dataset; "
        f"available X ids: {sorted(have_xids, key=int)}")

    pool = ThreadPoolExecutor(max_workers=16)
    # interleave configs so an early budget stop still covers many reciters
    order = []
    rnd = random.Random(7)
    cfgs = sorted(cfg_files)
    rnd.shuffle(cfgs)
    for cfg in cfgs:
        for p in sorted(cfg_files[cfg]):
            order.append((cfg, p))

    cols = ["audio", "duration_seconds", "phonemes", "match_ratio",
            "has_istiaatha", "has_bismillah", "has_sadaka", "segment_index",
            "reciter_english_name"]

    def handle(cfg, path):
        if path in st["done_files"]:
            return
        xid = cfg.split("_")[1].split(".")[0]
        is_dev = xid in dev_xids
        if is_dev and st["unseen_sec"] >= 2.5 * 3600:
            return
        if not is_dev and st["train_sec"] >= args.target_hours * 3600:
            return
        if not is_dev and st["cfg_sec"].get(cfg, 0.0) >= args.cap_hours * 3600:
            return
        for attempt in range(4):
            try:
                fh = fs.open(path, "rb")
                pf = pq.ParquetFile(fh)
                break
            except Exception as e:
                print(f"[mu] open retry {attempt} {path.rsplit('/', 1)[-1]}: {e}")
                if attempt == 3:
                    return
        if "phonemes" not in [f.name for f in pf.schema_arrow]:
            st["done_files"].append(path)
            save_state(state_path, st)
            return
        snap = json.loads(json.dumps(st))  # restored on mid-file failure
        futures = []
        try:
            for batch in pf.iter_batches(batch_size=24, columns=cols):
                names = batch.schema.names
                get = lambda c: batch.column(names.index(c)).to_pylist()
                for audio, dur, ph, ratio, ist, bsm, sdk, seg, rec_name in zip(
                        get("audio"), get("duration_seconds"), get("phonemes"),
                        get("match_ratio"), get("has_istiaatha"), get("has_bismillah"),
                        get("has_sadaka"), get("segment_index"), get("reciter_english_name")):
                    dur = float(dur or 0)
                    if dur < DUR_MIN or dur > DUR_MAX:
                        st["skip"]["dur"] += 1
                        continue
                    if sdk:
                        st["skip"]["sadaka"] += 1
                        continue
                    if ratio is None or float(ratio) < args.match_ratio_min:
                        st["skip"]["ratio"] += 1
                        continue
                    if not ph:
                        continue
                    raw = (audio or {}).get("bytes")
                    if not raw:
                        continue
                    try:
                        units = chunck_phonemes(ph)
                    except Exception:
                        st["oov_rows"] += 1
                        continue
                    if any(u not in vocab for u in units) or not units:
                        st["oov_rows"] += 1
                        continue
                    if is_dev:
                        if st["unseen_sec"] >= 2.5 * 3600:
                            break
                        st["unseen_sec"] += dur
                        split = "mu_unseen"
                    else:
                        if st["train_sec"] >= args.target_hours * 3600 or \
                           st["cfg_sec"].get(cfg, 0.0) >= args.cap_hours * 3600:
                            break
                        st["cfg_sec"][cfg] = st["cfg_sec"].get(cfg, 0.0) + dur
                        st["train_sec"] += dur
                        split = "mu_train"
                    # shard stem included: segment_index uniqueness across a
                    # config's shards is unverified — never risk overwrites
                    stem = os.path.basename(path).rsplit(".", 1)[0].split("-of-")[0]
                    name = (f"{'muunseen' if is_dev else 'mutrain'}_{cfg}_{stem}_{seg}.wav"
                            .replace("/", "_"))
                    wav_path = os.path.join(args.wavs, name)
                    meta = dict(id=name[:-4], wav=wav_path, dur=dur,
                                units=" ".join(units), vkey="",
                                reciter=f"mu_{xid}", split=split,
                                prefix=bool(ist or bsm))
                    futures.append((pool.submit(ffmpeg_decode, raw, wav_path), meta))
        except Exception as e:
            print(f"[mu] file read failed mid-stream, will retry on next paste: {e}")
            for fut, row in futures:
                fut.result()
            # roll back this file's counter mutations (file is NOT marked done)
            for k in ("cfg_sec", "train_sec", "unseen_sec", "oov_rows", "skip"):
                st[k] = snap[k]
            return
        ok = 0
        for fut, row in futures:
            if fut.result():
                rows.append(row)
                rows_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                ok += 1
        rows_f.flush()
        st["done_files"].append(path)
        save_state(state_path, st)
        print(f"[mu] {cfg} {path.rsplit('/', 1)[-1]}: +{ok} | train={st['train_sec'] / 3600:.1f}h "
              f"unseen={st['unseen_sec'] / 3600:.1f}h oov={st['oov_rows']} skip={st['skip']}")

    for cfg, p in order:
        handle(cfg, p)
    pool.shutdown(wait=True)
    save_state(state_path, st)

    # dedupe sidecar rows from any mid-file crash/retry
    rows = list({r["id"]: r for r in rows}.values())

    assert st["oov_rows"] == 0, (
        f"{st['oov_rows']} muaalem rows had out-of-vocab units — the gold labels "
        "no longer match our 250-unit scheme; DO NOT TRAIN, investigate first")

    train_rows = [r for r in rows if r["split"] == "mu_train"]
    n, sec = rows_to_cuts(train_rows, os.path.join(args.out, "mu_cuts_train_raw.jsonl.gz"))
    print(f"[mu] train cuts: {n} ({sec / 3600:.1f} h)")
    unseen_rows = [r for r in rows if r["split"] == "mu_unseen"]
    write_jsonl(os.path.join(args.evalsets, "mu_unseen.jsonl"), unseen_rows)
    print(f"[mu] unseen eval: {len(unseen_rows)} clips "
          f"({st['unseen_sec'] / 3600:.1f} h) from configs X in {sorted(dev_xids)}")


if __name__ == "__main__":
    main()
PYEOF

cat > "$TOOLS/prepare_retasy.py" <<'PYEOF'
"""RetaSy/quranic_audio_dataset -> EVAL-ONLY real-user sets.

1,287 non-professional learners (81 countries) reciting verses on their own
devices — the closest public proxy for Noor's real users. Not used for
training (only ~470 crowd-verified-correct clips): kept fully held out.

  rs_user.jsonl     final_label == "correct" or golden — headline eval set
  rs_mistake.jsonl  final_label == "in_correct" — real mistakes; a model that
                    "auto-corrects" them to the canonical text is FAILING
                    (PER vs canonical must stay clearly above rs_user PER)

Refs = canonical units matched from the Aya text (two-tier matcher);
ayah-1 rows skipped (same basmala ambiguity as everyayah).
"""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor

from prep_common import (build_matchers, ffmpeg_decode, match_verse,
                         write_jsonl)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--wavs", required=True)
    ap.add_argument("--evalsets", required=True)
    args = ap.parse_args()

    from huggingface_hub import HfFileSystem
    import pyarrow.parquet as pq

    labels = json.load(open(args.labels, encoding="utf-8"))
    matchers = build_matchers(labels)
    tier_hits = {"mild": 0, "alif": 0, "rasm": 0, "miss": 0}

    os.makedirs(args.wavs, exist_ok=True)
    fs = HfFileSystem()
    files = sorted(fs.glob("datasets/RetaSy/quranic_audio_dataset/**/*.parquet"))
    assert files, "RetaSy parquet not found"
    pool = ThreadPoolExecutor(max_workers=16)

    user_rows, mistake_rows = [], []
    futures = []
    for path in files:
        fh = fs.open(path, "rb")
        pf = pq.ParquetFile(fh)
        cols = ["audio", "Aya", "duration_ms", "golden", "final_label",
                "reciter_id", "reciter_qiraah"]
        avail = [f.name for f in pf.schema_arrow]
        cols = [c for c in cols if c in avail]
        for batch in pf.iter_batches(batch_size=32, columns=cols):
            names = batch.schema.names
            get = lambda c: batch.column(names.index(c)).to_pylist()
            for audio, aya, dur_ms, golden, label, rid, qiraah in zip(
                    get("audio"), get("Aya"), get("duration_ms"),
                    get("golden"), get("final_label"), get("reciter_id"),
                    get("reciter_qiraah")):
                # refs are canonical HAFS units — a different qiraah would be
                # scored as "errors"; ~960/6828 clips are non-hafs/unknown
                if (qiraah or "").strip().lower() != "hafs":
                    continue
                label = (label or "").strip()
                is_user = bool(golden) or label == "correct"
                is_mistake = label == "in_correct"
                if not (is_user or is_mistake):
                    continue
                dur = float(dur_ms or 0) / 1000.0
                if dur < 1.0 or dur > 30.0:
                    continue
                hit = match_verse(aya or "", matchers, tier_hits)
                if hit is None:
                    continue
                vkey, units, _ = hit
                if vkey.split(":")[1] == "1":
                    continue
                raw = (audio or {}).get("bytes")
                if not raw:
                    continue
                kind = "user" if is_user else "mistake"
                name = f"rs_{kind}_{len(futures):05d}_{vkey.replace(':', '_')}.wav"
                wav_path = os.path.join(args.wavs, name)
                meta = dict(id=name[:-4], wav=wav_path, dur=dur, units=units,
                            vkey=vkey, reciter=str(rid), split=f"rs_{kind}")
                futures.append((pool.submit(ffmpeg_decode, raw, wav_path), meta))
    for fut, row in futures:
        if fut.result():
            (user_rows if row["split"] == "rs_user" else mistake_rows).append(row)
    pool.shutdown(wait=True)

    write_jsonl(os.path.join(args.evalsets, "rs_user.jsonl"), user_rows)
    write_jsonl(os.path.join(args.evalsets, "rs_mistake.jsonl"), mistake_rows)
    speakers = len({r["reciter"] for r in user_rows})
    print(f"[rs] rs_user: {len(user_rows)} clips / {speakers} speakers; "
          f"rs_mistake: {len(mistake_rows)} clips; tiers={tier_hits}")
    # floors set from the dataset's REAL yield (409 correct + 62 golden total,
    # minus non-hafs, ayah-1, fragments/isti'adha rows that legitimately miss)
    assert len(user_rows) >= 120, "too few verified-correct RetaSy clips — investigate"
    assert len(mistake_rows) >= 120, "too few RetaSy mistake clips — investigate"


if __name__ == "__main__":
    main()
PYEOF

cat > "$TOOLS/make_features.py" <<'PYEOF'
"""Per-source features (v2): speed-perturb x3 + RIR reverb view on a seeded
subset + 80-dim fbank. The manifest is SHUFFLED before writing (the v1
manifest was reciter-and-speed ordered; the sampler's default buffer only
shuffles ~4% of it). MUSAN/RIRS pools are built with an eval holdout."""
import argparse
import glob
import os
import random
import shutil
import time

import torch

# lhotse's process-pool feature workers deadlock when torch multithreads
# inside each worker. One thread per worker; parallelism = --num-jobs.
torch.set_num_threads(1)

from dataclasses import replace as dc_replace

from lhotse import (CutSet, Fbank, FbankConfig, LilcomChunkyWriter, Recording,
                    RecordingSet)
from lhotse.cut import MonoCut
from lhotse.utils import fastcopy


def n_jobs():
    try:
        n = len(os.sched_getaffinity(0))
    except AttributeError:
        n = os.cpu_count() or 8
    return max(2, min(n, 16))


def compute_local_then_copy(cuts, extractor, local_name, final_store):
    """The network volume throws EIO under sustained many-writer load, but a
    single sequential bulk copy usually survives (and is cheaply retried).
    Compute features on the pod-local disk, then copy once to the volume and
    rewrite the manifest's storage paths."""
    local = os.path.join("/tmp", local_name)
    shutil.rmtree(local, ignore_errors=True)
    cs = cuts.compute_and_store_features(
        extractor=extractor, storage_path=local,
        num_jobs=n_jobs(), storage_type=LilcomChunkyWriter)
    for attempt in range(5):
        try:
            shutil.rmtree(final_store, ignore_errors=True)
            shutil.copytree(local, final_store)
            break
        except OSError as e:
            print(f"[feats] volume copy attempt {attempt + 1} I/O error: {e} — retrying in 30s")
            time.sleep(30)
    else:
        raise SystemExit("bulk copy to volume failed 5 attempts — stop/start the pod, re-paste")
    shutil.rmtree(local, ignore_errors=True)
    return CutSet.from_cuts(
        fastcopy(c, features=dc_replace(
            c.features, storage_path=c.features.storage_path.replace(local, final_store)))
        for c in cs).to_eager()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--feats", required=True)
    ap.add_argument("--source", required=True, help="ea | mu | indev")
    ap.add_argument("--rirs-dir", default="")
    ap.add_argument("--rir-fraction", type=float, default=0.30)
    ap.add_argument("--musan-dir", default="")
    args = ap.parse_args()
    extractor = Fbank(FbankConfig(num_mel_bins=80))

    if args.source in ("ea", "mu"):
        src = args.source
        out_manifest = os.path.join(args.data, f"{src}_cuts_train.jsonl.gz")
        if not os.path.exists(out_manifest):
            cuts = CutSet.from_file(os.path.join(args.data, f"{src}_cuts_train_raw.jsonl.gz"))
            cuts = cuts + cuts.perturb_speed(0.9) + cuts.perturb_speed(1.1)
            cuts = cuts.to_eager()
            if args.rirs_dir:
                rirs = sorted(
                    glob.glob(os.path.join(args.rirs_dir, "simulated_rirs", "**", "*.wav"),
                              recursive=True)
                    + [p for p in glob.glob(os.path.join(
                        args.rirs_dir, "real_rirs_isotropic_noises", "*.wav"))
                       if "rir" in os.path.basename(p).lower()])
                assert len(rirs) >= 100, (
                    f"only {len(rirs)} RIRs under {args.rirs_dir} — unzip layout changed?")
                # hold out every 10th RIR for eval-time reverb
                train_rirs = [p for i, p in enumerate(rirs) if i % 10 != 0]
                eval_rirs = [p for i, p in enumerate(rirs) if i % 10 == 0]
                with open(os.path.join(args.data, "eval_rirs.txt"), "w") as f:
                    f.write("\n".join(eval_rirs))
                rir_recs = RecordingSet.from_recordings(
                    Recording.from_file(p) for p in train_rirs)
                rnd = random.Random(1234)
                ids = [c.id for c in cuts]
                pick = set(rnd.sample(ids, int(len(ids) * args.rir_fraction)))
                wet = cuts.filter(lambda c: c.id in pick).to_eager()
                wet = wet.reverb_rir(rir_recordings=rir_recs, normalize_output=True)
                wet = wet.modify_ids(lambda cid: cid + "_rir")
                cuts = (cuts + wet).to_eager()
                print(f"[feats:{src}] +{len(wet)} reverberated views "
                      f"({len(train_rirs)} train RIRs, {len(eval_rirs)} held for eval)")
            cuts = cuts.shuffle(rng=random.Random(4242))
            # RESUMABLE SLICES: the network volume throws sporadic EIO under
            # sustained writes. One monolithic compute loses hours per hiccup;
            # a slice loses minutes and completed slices are never redone.
            # The view construction above is fully deterministic (seeded), so
            # slice membership is identical across re-pastes.
            all_cuts = list(cuts)
            n_slices = 24
            slice_manifests = []
            for si in range(n_slices):
                sl = all_cuts[si::n_slices]
                sm = os.path.join(args.data, f"{src}_feats_slice_{si:02d}.jsonl.gz")
                slice_manifests.append(sm)
                if os.path.exists(sm):
                    print(f"[feats:{src}] slice {si + 1}/{n_slices} already done")
                    continue
                storage = os.path.join(args.feats, f"quran_{src}_s{si:02d}")
                for attempt in range(3):
                    try:
                        shutil.rmtree(storage, ignore_errors=True)
                        sub = CutSet.from_cuts(sl).compute_and_store_features(
                            extractor=extractor,
                            storage_path=storage,
                            num_jobs=n_jobs(),
                            storage_type=LilcomChunkyWriter,
                        )
                        sub.to_file(sm)
                        break
                    except OSError as e:
                        print(f"[feats:{src}] slice {si + 1} attempt {attempt + 1} "
                              f"I/O error: {e} — retrying in 30s")
                        time.sleep(30)
                else:
                    raise SystemExit(
                        f"slice {si + 1} failed 3 attempts — volume unhealthy: "
                        "STOP and START the pod (fresh mount), then re-paste")
                print(f"[feats:{src}] slice {si + 1}/{n_slices} done ({len(sl)} cuts)")
            # leftovers from the abandoned monolithic attempt waste quota
            shutil.rmtree(os.path.join(args.feats, f"quran_{src}"), ignore_errors=True)
            merged = CutSet.from_cuts(
                c for sm in slice_manifests for c in CutSet.from_file(sm))
            merged.to_file(out_manifest)
            print(f"[feats:{src}] {len(merged)} cuts with features")
        else:
            print(f"[feats:{src}] already done")
        # verification gate before the caller may delete this source's wavs
        check = CutSet.from_file(out_manifest)
        sample = [c for _, c in zip(range(5), check)]
        assert len(sample) == 5, f"{src} train manifest has <5 cuts"
        for c in sample:
            f = c.load_features()
            assert f is not None and f.shape[1] == 80, "feature load failed"
        print(f"[feats:{src}] VERIFIED (sampled feature loads OK)")
        return

    if args.source == "indev":
        out_manifest = os.path.join(args.data, "quran_cuts_dev.jsonl.gz")
        if os.path.exists(out_manifest):
            print("[feats:indev] already done")
            return
        cuts = CutSet.from_file(os.path.join(args.data, "ea_cuts_indev_raw.jsonl.gz"))
        cuts_f = compute_local_then_copy(
            cuts, extractor, "noor_feats_dev", os.path.join(args.feats, "quran_dev"))
        assert len(cuts_f) > 0, "dev set is EMPTY — refusing (an empty dev reports a perfect score)"
        cuts_f.to_file(out_manifest)
        print(f"[feats:indev] {len(cuts_f)} dev cuts with features")
        return

    if args.source == "musan":
        out_manifest = os.path.join(args.data, "musan_cuts.jsonl.gz")
        if os.path.exists(out_manifest):
            print("[feats] musan already done")
            return
        wavs = sorted(
            glob.glob(os.path.join(args.musan_dir, "music", "**", "*.wav"), recursive=True)
            + glob.glob(os.path.join(args.musan_dir, "noise", "**", "*.wav"), recursive=True))
        if args.rirs_dir:
            wavs += sorted(
                p for p in glob.glob(os.path.join(
                    args.rirs_dir, "real_rirs_isotropic_noises", "*.wav"))
                if "noise" in os.path.basename(p).lower())
        # every 10th file is held out for EVAL noise (never mixed in training)
        train_wavs = [p for i, p in enumerate(wavs) if i % 10 != 0]
        eval_wavs = [p for i, p in enumerate(wavs) if i % 10 == 0]
        with open(os.path.join(args.data, "eval_noises.txt"), "w") as f:
            f.write("\n".join(eval_wavs))
        print(f"[feats] noise pool: {len(train_wavs)} train / {len(eval_wavs)} eval-holdout")
        cuts = []
        for w in train_wavs:
            rid = os.path.splitext(os.path.basename(w))[0]
            try:
                rec = Recording.from_file(w, recording_id=rid)
            except Exception:
                continue
            cuts.append(MonoCut(id=rid, start=0.0, duration=rec.duration,
                                channel=0, recording=rec))
        musan = CutSet.from_cuts(cuts).cut_into_windows(duration=10.0)
        # windowing leaves sub-frame tail slivers that crash the fbank layer
        musan = musan.filter(lambda c: c.duration >= 0.5).to_eager()
        musan_f = compute_local_then_copy(
            musan, extractor, "noor_feats_musan", os.path.join(args.feats, "musan"))
        musan_f.to_file(out_manifest)
        print(f"[feats] musan: {len(musan_f)} cuts with features")
        return

    raise SystemExit(f"unknown --source {args.source}")


if __name__ == "__main__":
    main()
PYEOF

cat > "$TOOLS/cleanup_train_wavs.py" <<'PYEOF'
"""Delete exactly the wavs referenced by a source's RAW TRAIN manifest (its
features are computed and verified by then), then assert every wav referenced
by the given eval jsonls still exists. Manifest-driven: immune to filename
prefixes disagreeing with re-derived splits."""
import json
import os
import sys

from lhotse import CutSet

manifest, evaljsonls = sys.argv[1], sys.argv[2:]
n = 0
for c in CutSet.from_file(manifest):
    p = c.recording.sources[0].source
    if os.path.exists(p):
        os.remove(p)
        n += 1
print(f"[cleanup] deleted {n} train wavs listed in {os.path.basename(manifest)}")
missing = 0
for ej in evaljsonls:
    with open(ej, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not os.path.exists(json.loads(line)["wav"]):
                missing += 1
assert missing == 0, f"{missing} eval wavs went missing — eval sets damaged, investigate before continuing"
print("[cleanup] eval wavs intact")
PYEOF

cat > "$TOOLS/merge_train.py" <<'PYEOF'
"""Merge the per-source train manifests into the final shuffled train manifest
and print the final hour count."""
import os
import random

from lhotse import CutSet

data = os.environ["DATA"]
sources = {}
for src, env in (("ea", "EA_TRAIN_HOURS"), ("mu", "MU_TRAIN_HOURS")):
    cs = CutSet.from_file(os.path.join(data, f"{src}_cuts_train.jsonl.gz")).to_eager()
    h = sum(c.duration for c in cs) / 3600
    # raw target x3 speed views (+~30% RIR) — floor at 55% of the x3 figure
    floor = 0.55 * float(os.environ[env]) * 3
    print(f"[merge] {src}: {len(cs)} cuts, {h:.0f} effective hours (floor {floor:.0f})")
    assert h >= floor, f"{src} contributes only {h:.0f}h effective — a source went missing, do not train"
    sources[src] = cs
merged = (sources["ea"] + sources["mu"]).to_eager().shuffle(rng=random.Random(20260815))
out = os.path.join(data, "quran_cuts_train.jsonl.gz")
merged.to_file(out)
hours = sum(c.duration for c in merged) / 3600
print(f"[merge] final train manifest: {len(merged)} cuts, {hours:.0f} effective hours -> {out}")
PYEOF

cat > "$TOOLS/asr_datamodule.py" <<'PYEOF'
"""Quran data module exposing the exact class/API icefall's zipformer train.py
imports (LibriSpeechAsrDataModule). v2 changes vs v1:
- DynamicBucketingSampler gets buffer_size=num_buckets*5000 (stock icefall
  behavior; the default 20k buffer shuffled ~4% of our manifest).
- CutMix snr widened to (5,25) — phone mics see worse SNR than (10,20).
- MUSAN manifest loaded EAGERLY (CutMix needs random access; lazy iteration
  correlates noise choice with batch order).
- GainJitter input transform: per-utterance +/-10 dB shift applied to the
  log-mel features (log-domain add — zero I/O). The app feeds UN-normalized
  mic audio, so absolute level robustness must come from training.
"""
import argparse
import logging
import math
import random
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from lhotse import CutSet, Fbank, FbankConfig, load_manifest, load_manifest_lazy
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


class GainJitter:
    """Per-utterance gain jitter in the log-mel domain.

    Multiplying the waveform by g multiplies mel POWER by g^2, i.e. adds
    2*ln(g) to every bin of lhotse's natural-log fbank. For a gain of D dB,
    2*ln(g) = D * ln(10) / 10. Applied before SpecAugment.
    """

    def __init__(self, min_db: float = -10.0, max_db: float = 10.0, p: float = 0.8):
        self.min_db = min_db
        self.max_db = max_db
        self.p = p

    def __call__(self, features: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        for i in range(features.size(0)):
            if random.random() < self.p:
                db = random.uniform(self.min_db, self.max_db)
                features[i] += db * math.log(10.0) / 10.0
        return features


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
            logging.info("Enable MUSAN (snr 5..25, eager manifest)")
            cuts_musan = load_manifest(self.args.manifest_dir / "musan_cuts.jsonl.gz")
            transforms.append(CutMix(cuts=cuts_musan, p=0.5, snr=(5, 25), preserve_id=True))
        else:
            logging.info("Disable MUSAN")

        if self.args.concatenate_cuts:
            transforms = [
                CutConcatenate(duration_factor=self.args.duration_factor, gap=self.args.gap)
            ] + transforms

        input_transforms = [GainJitter(min_db=-10.0, max_db=10.0, p=0.8)]
        logging.info("Enable GainJitter +/-10dB p=0.8")
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
                buffer_size=self.args.num_buckets * 5000,
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

cat > "$TOOLS/eval_matrix.py" <<'PYEOF'
"""v2 eval: unit-PER (primary) + char-PER for every model x set x condition.

Sets (jsonl: {wav, units, ...}):
  ea_indev    in-reciter everyayah dev          clean / snr15 / snr5
  ea_unseen   held-out everyayah reciters       clean / snr15 / snr5 / reverb
  mu_unseen   held-out muaalem reciters         clean / snr15
  rs_user     RetaSy verified-correct learners  clean / snr15
  rs_mistake  RetaSy real mistakes              clean   (PER must stay HIGH)
Probes (derived from ea_unseen):
  probe_trunc audio cut at 60% — hyp must STOP (len ratio vs 0.6*ref)
  probe_cross refs shifted by one — PER must stay HIGH (no verse prior)

Noise/reverb come from the held-out eval pools (never seen in training),
seeded per utterance id -> reproducible across models.
"""
import argparse
import json
import os
import random
import re

import numpy as np
import soundfile as sf


def read_jsonl(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


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


_RUN_CAP = re.compile(r"(.)\1{2,}")


def cap_runs(s):
    """The app matcher caps character runs at 2 (normalizePhonemesForMatch) so
    madd/ghunnah LENGTH never counts as an error. The matcher-faithful metric
    (cap_char_per) applies the same normalization to ref and hyp — without it,
    canonical madd-4 refs would punish correct-but-short-madd recitation,
    exactly the profile of real learners in rs_user."""
    return _RUN_CAP.sub(r"\1\1", s)


class Unitizer:
    """Greedy longest-match re-tokenizer over the 250-unit vocab. Applied
    identically to ref and hyp so unit-PER is self-consistent."""

    def __init__(self, tokens_path):
        self.units = set()
        with open(tokens_path, encoding="utf-8") as f:
            for line in f:
                u = line.rstrip("\n").rsplit(" ", 1)[0]
                if u != "<blk>":
                    self.units.add(u)
        self.max_len = max(len(u) for u in self.units)

    def __call__(self, s):
        out, i = [], 0
        while i < len(s):
            for ln in range(min(self.max_len, len(s) - i), 0, -1):
                if s[i:i + ln] in self.units:
                    out.append(s[i:i + ln])
                    i += ln
                    break
            else:
                out.append(s[i])  # stray char = its own (wrong) token
                i += 1
        return out


def seeded_rng(uid):
    import hashlib
    return np.random.default_rng(int(hashlib.sha1(uid.encode()).hexdigest()[:8], 16))


def load_wav(path):
    samples, sr = sf.read(path, dtype="float32")
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    return samples, sr


def add_noise(samples, noise, snr_db):
    n = noise
    while len(n) < len(samples):
        n = np.concatenate([n, noise])
    n = n[: len(samples)]
    sp = np.sqrt((samples ** 2).mean() + 1e-9)
    npow = np.sqrt((n ** 2).mean() + 1e-9)
    n = n * (sp / npow) * (10 ** (-snr_db / 20))
    return np.clip(samples + n, -1.0, 1.0)


def add_reverb(samples, rir):
    # peak-align the RIR so the direct path lands at t=0 (keeps alignment)
    k = np.argmax(np.abs(rir))
    rir = rir[k:]
    rir = rir / (np.max(np.abs(rir)) + 1e-9)
    n = len(samples) + len(rir) - 1
    nfft = 1 << (n - 1).bit_length()
    out = np.fft.irfft(np.fft.rfft(samples, nfft) * np.fft.rfft(rir, nfft))[: len(samples)]
    peak = np.max(np.abs(out)) + 1e-9
    ref = np.max(np.abs(samples)) + 1e-9
    return np.clip(out * (ref / peak), -1.0, 1.0).astype("float32")


def decode(rec, samples, sr):
    stream = rec.create_stream()
    step = 1600
    for off in range(0, len(samples), step):
        stream.accept_waveform(sr, samples[off: off + step])
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
    ap.add_argument("--models", required=True,
                    help="comma list of name=path (all share --tokens)")
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--evalsets", required=True)
    ap.add_argument("--eval-noises", required=True)
    ap.add_argument("--eval-rirs", required=True)
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--report", required=True)
    args = ap.parse_args()

    import sherpa_onnx

    unitize = Unitizer(args.tokens)
    noise_paths = [l.strip() for l in open(args.eval_noises) if l.strip()]
    rir_paths = [l.strip() for l in open(args.eval_rirs) if l.strip()]
    assert noise_paths and rir_paths, "eval noise/RIR holdout pools are empty — rerun S5a/S5c"

    def load_sets():
        sets = {}
        for name, limit in (("ea_indev", args.limit), ("ea_unseen", args.limit + 100),
                            ("mu_unseen", args.limit), ("rs_user", args.limit + 200),
                            ("rs_mistake", args.limit)):
            rows = read_jsonl(os.path.join(args.evalsets, f"{name}.jsonl"))
            random.Random(777).shuffle(rows)
            sets[name] = rows[:limit]
            assert sets[name], f"eval set {name} is EMPTY — refusing to report a perfect score"
        return sets

    sets = load_sets()
    conditions = {
        "ea_indev": ["clean", "snr15", "snr5"],
        "ea_unseen": ["clean", "snr15", "snr5", "reverb"],
        "mu_unseen": ["clean", "snr15"],
        "rs_user": ["clean", "snr15"],
        "rs_mistake": ["clean"],
    }

    report = {}
    for spec in args.models.split(","):
        mname, mpath = spec.split("=", 1)
        print(f"[eval] === model {mname}: {mpath}")
        rec = sherpa_onnx.OnlineRecognizer.from_zipformer2_ctc(
            tokens=args.tokens, model=mpath, num_threads=4)
        mreport = {}
        for sname, rows in sets.items():
            for cond in conditions[sname]:
                errs_u = tot_u = errs_c = tot_c = errs_m = tot_m = 0
                scored = skipped = 0
                for r in rows:
                    # one corrupt wav or transient volume read error must not
                    # kill a 2-hour stage — skip, count, and gate on the count
                    try:
                        samples, sr = load_wav(r["wav"])
                        rng = seeded_rng(r["id"] + cond)
                        if cond.startswith("snr"):
                            noise, _ = load_wav(noise_paths[int(rng.integers(len(noise_paths)))])
                            samples = add_noise(samples, noise, float(cond[3:]))
                        elif cond == "reverb":
                            rir, _ = load_wav(rir_paths[int(rng.integers(len(rir_paths)))])
                            samples = add_reverb(samples, rir)
                        hyp = decode(rec, samples, sr)
                    except Exception as e:
                        skipped += 1
                        print(f"[eval] skip {r['id']} ({cond}): {e}")
                        continue
                    scored += 1
                    ref = r["units"].replace(" ", "")
                    ru, hu = unitize(ref), unitize(hyp)
                    errs_u += edit_distance(ru, hu); tot_u += len(ru)
                    errs_c += edit_distance(ref, hyp); tot_c += len(ref)
                    refm, hypm = cap_runs(ref), cap_runs(hyp)
                    errs_m += edit_distance(refm, hypm); tot_m += len(refm)
                assert skipped <= max(3, len(rows) // 10), (
                    f"{skipped}/{len(rows)} clips unreadable in {sname}/{cond} — volume unhealthy, "
                    "stop/start the pod and re-paste")
                mreport.setdefault(sname, {})[cond] = {
                    "unit_per": round(100.0 * errs_u / max(1, tot_u), 2),
                    "char_per": round(100.0 * errs_c / max(1, tot_c), 2),
                    "cap_char_per": round(100.0 * errs_m / max(1, tot_m), 2),
                    "clips": scored,
                }
                v = mreport[sname][cond]
                print(f"[eval] {mname} {sname}/{cond}: "
                      f"unit-PER {v['unit_per']}% cap-char-PER {v['cap_char_per']}% "
                      f"({scored} clips, {skipped} skipped)")

        # probes from ea_unseen
        probe_rows = sets["ea_unseen"][:150]
        # probe_trunc: cut audio at 60%; a verse-prior model completes anyway
        ratios = []
        for r in probe_rows:
            try:
                samples, sr = load_wav(r["wav"])
                hyp = decode(rec, samples[: int(len(samples) * 0.6)], sr)
            except Exception as e:
                print(f"[eval] skip probe_trunc {r['id']}: {e}")
                continue
            ref_u = unitize(r["units"].replace(" ", ""))
            exp_len = max(1.0, 0.6 * len(ref_u))
            ratios.append(len(unitize(hyp)) / exp_len)
        frac_ok = sum(1 for x in ratios if x <= 1.15) / max(1, len(ratios))
        mreport["probe_trunc"] = {"clean": {
            "len_ratio_mean": round(float(np.mean(ratios)), 3),
            "frac_within_115": round(frac_ok, 3), "clips": len(ratios)}}
        print(f"[eval] {mname} probe_trunc: mean len ratio "
              f"{mreport['probe_trunc']['clean']['len_ratio_mean']} "
              f"(within 1.15x: {frac_ok:.0%})")
        # probe_cross: score audio i against ref i+1 — PER must stay HIGH
        errs = tot = 0
        for i, r in enumerate(probe_rows[:100]):
            other = probe_rows[(i + 1) % len(probe_rows)]
            try:
                samples, sr = load_wav(r["wav"])
                hyp = decode(rec, samples, sr)
            except Exception as e:
                print(f"[eval] skip probe_cross {r['id']}: {e}")
                continue
            wrong_ref = unitize(other["units"].replace(" ", ""))
            errs += edit_distance(wrong_ref, unitize(hyp)); tot += len(wrong_ref)
        mreport["probe_cross"] = {"clean": {
            "unit_per": round(100.0 * errs / max(1, tot), 2), "clips": 100}}
        print(f"[eval] {mname} probe_cross: unit-PER "
              f"{mreport['probe_cross']['clean']['unit_per']}% (must stay HIGH)")
        report[mname] = mreport
        del rec

    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    print(f"[eval] matrix -> {args.report}")


if __name__ == "__main__":
    main()
PYEOF

cat > "$TOOLS/gate.py" <<'PYEOF'
"""Ship gates over the eval matrix. Exit non-zero (and the driver refuses to
populate $OUT) unless the candidate beats the incumbent and passes the
absolute + mistake-fitness bars."""
import argparse
import json
import sys

G2_EA_UNSEEN_MAX = 12.0    # unit-PER %, clean (Muno v1 analog: 11.63)
G2_RS_USER_MAX = 16.0      # unit-PER %, clean (Tarteel cloud: 16.2 WER)
G3_INT8_DELTA_MAX = 1.0    # abs points vs fp32
G4_CROSS_MIN = 60.0        # probe_cross unit-PER must stay above
G4_TRUNC_FRAC = 0.90       # fraction of trunc clips within 1.15x
# real user mistakes are often single-phoneme deviations, so their PER may sit
# only modestly above rs_user even for a prior-free model — 1.3, not higher
G4_MISTAKE_FACTOR = 1.3    # rs_mistake PER >= factor * rs_user PER


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", required=True)
    ap.add_argument("--incumbent", default="shipped")
    ap.add_argument("--candidate", default="new_int8_ch8")
    ap.add_argument("--fp32", default="new_fp32_ch8")
    ap.add_argument("--report", required=True)
    args = ap.parse_args()

    m = json.load(open(args.matrix, encoding="utf-8"))
    inc, cand, fp32 = m[args.incumbent], m[args.candidate], m.get(args.fp32)
    results, ok = [], True

    def check(name, passed, detail):
        nonlocal ok
        results.append({"gate": name, "passed": bool(passed), "detail": detail})
        ok = ok and passed
        print(f"[gate] {'PASS' if passed else 'FAIL'} {name}: {detail}")

    # The matcher-faithful metric (cap_char_per: character runs capped at 2,
    # exactly what the app's normalizePhonemesForMatch does) drives the gates.
    # G1 compares ONLY on the FAIR sets. The everyayah sets are TRAINING DATA
    # for the incumbent (v1 trained on those exact recordings, no reciter
    # holdout — measured 0.3-0.6% PER there = memorization, not skill), so a
    # beats-incumbent test on them would fail any honestly-held-out model.
    # Fair: mu_unseen (v1 never saw muaalem) + rs_user (real users, new to both).
    # rs_mistake is EXCLUDED: its PER is SUPPOSED to be high (G4 gates it the
    # other way).
    FAIR_SETS = ("mu_unseen", "rs_user")
    worst = []
    for sname in FAIR_SETS:
        for cond, v in cand.get(sname, {}).items():
            iv = inc.get(sname, {}).get(cond)
            if iv is None or "cap_char_per" not in v:
                continue
            if v["cap_char_per"] >= iv["cap_char_per"]:
                worst.append(f"{sname}/{cond}: {v['cap_char_per']} vs {iv['cap_char_per']}")
    check("G1a beats incumbent on fair sets (cap-char)", not worst,
          worst or "all better")
    c = cand["rs_user"]["clean"]["cap_char_per"]
    i = inc["rs_user"]["clean"]["cap_char_per"]
    check("G1b rs_user <= 0.6x incumbent (cap-char)", c <= 0.6 * i,
          f"{c} vs 0.6*{i}={0.6 * i:.1f}")

    # G2: absolute bars — ea_unseen on unit-PER (comparable to the reference
    # model's published unseen-reciter 11.63%), rs_user on the matcher metric
    check("G2 ea_unseen clean <= 12% unit-PER",
          cand["ea_unseen"]["clean"]["unit_per"] <= G2_EA_UNSEEN_MAX,
          cand["ea_unseen"]["clean"]["unit_per"])
    check("G2 rs_user clean <= 16% cap-char-PER",
          cand["rs_user"]["clean"]["cap_char_per"] <= G2_RS_USER_MAX,
          cand["rs_user"]["clean"]["cap_char_per"])

    # G3: int8 quantization damage
    if fp32:
        deltas = []
        for sname, conds in cand.items():
            if sname.startswith("probe"):
                continue
            for cond, v in conds.items():
                fv = fp32.get(sname, {}).get(cond)
                if fv and "unit_per" in v:
                    deltas.append((f"{sname}/{cond}", v["unit_per"] - fv["unit_per"]))
        bad = [(k, round(d, 2)) for k, d in deltas if d > G3_INT8_DELTA_MAX]
        check("G3 int8-fp32 <= 1.0 point everywhere", not bad, bad or "ok")

    # G4: mistake-detection fitness (no verse-prior autocorrect)
    check("G4 probe_cross >= 60%", cand["probe_cross"]["clean"]["unit_per"] >= G4_CROSS_MIN,
          cand["probe_cross"]["clean"]["unit_per"])
    check("G4 probe_trunc 90% within 1.15x",
          cand["probe_trunc"]["clean"]["frac_within_115"] >= G4_TRUNC_FRAC,
          cand["probe_trunc"]["clean"]["frac_within_115"])
    # capped metric here too: madd-length noise must not mask (or fake) the
    # separation between correct and mistaken recitation
    mistake = cand["rs_mistake"]["clean"]["cap_char_per"]
    user = cand["rs_user"]["clean"]["cap_char_per"]
    check(f"G4 rs_mistake >= {G4_MISTAKE_FACTOR}x rs_user (cap-char)",
          mistake >= G4_MISTAKE_FACTOR * user,
          f"{mistake} vs {G4_MISTAKE_FACTOR}*{user}={G4_MISTAKE_FACTOR * user:.1f}")

    with open(args.report, "w", encoding="utf-8") as f:
        json.dump({"passed": ok, "gates": results}, f, indent=1)
    print(f"[gate] {'ALL GATES PASSED' if ok else 'GATES FAILED'} -> {args.report}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
PYEOF

cat > "$TOOLS/check_prep.py" <<'PYEOF'
"""Prep gate (v2): per-source hour/count asserts BEFORE feature burn."""
import json
import os

base = os.environ["BASE"]
data = f"{base}/data"
evalsets = f"{base}/evalsets"


def jsonl_count(path):
    if not os.path.exists(path):
        return 0
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


# gate on the MANIFESTS, not the state counters: counters increment at row
# acceptance, before ffmpeg success/collisions — only the manifest is truth
from lhotse import CutSet

for src, env in (("ea", "EA_TRAIN_HOURS"), ("mu", "MU_TRAIN_HOURS")):
    target = float(os.environ[env])
    cs = CutSet.from_file(f"{data}/{src}_cuts_train_raw.jsonl.gz")
    h = sum(c.duration for c in cs) / 3600
    print(f"[gate] {src} raw train manifest: {len(cs)} cuts, {h:.1f}h (target {target:.0f}h)")
    assert h >= 0.55 * target, (
        f"{src} manifest only {h:.1f}h (<55% of {target:.0f}h) — re-paste to retry, "
        "or rows were lost between acceptance and decode")

ea = json.load(open(f"{data}/ea_state.json", encoding="utf-8"))
assert ea["indev_sec"] >= 0.5 * 3600, "in-reciter dev under 30 min"
mu = json.load(open(f"{data}/mu_state.json", encoding="utf-8"))
assert mu["oov_rows"] == 0, "muaalem OOV rows present"

counts = {}
for name, floor in (("ea_indev", 200), ("ea_unseen", 300), ("mu_unseen", 200),
                    ("rs_user", 120), ("rs_mistake", 120)):
    n = jsonl_count(f"{evalsets}/{name}.jsonl")
    counts[name] = n
    print(f"[gate] eval set {name}: {n} clips (floor {floor})")
    assert n >= floor, f"eval set {name} too small ({n} < {floor})"
print(f"[gate] PREP OK {counts}")
PYEOF

# ----------------------------- S4a: everyayah --------------------------------
if ! done_p s4a_everyayah; then
  log "S4a everyayah (target ${EA_TRAIN_HOURS}h train, cap ${EA_CAP_HOURS}h/reciter, held-out reciters via ${EA_DEV_RECITERS:-hash%$EA_DEV_MOD})"
  ( cd "$TOOLS" && $PY prepare_everyayah.py \
      --labels "$DATA/labels.json" --out "$DATA" --wavs "$WAVS" \
      --evalsets "$EVALSETS" \
      --target-hours "$EA_TRAIN_HOURS" --cap-hours "$EA_CAP_HOURS" \
      --indev-hours "$EA_INDEV_HOURS" --accept-p "$EA_ACCEPT_P" \
      --dev-mod "$EA_DEV_MOD" --dev-reciters "$EA_DEV_RECITERS" )
  mark s4a_everyayah
fi

# ----------------------------- S4b: muaalem ----------------------------------
if ! done_p s4b_muaalem; then
  log "S4b muaalem-annotated-v3 (target ${MU_TRAIN_HOURS}h, cap ${MU_CAP_HOURS}h/config, dev X in {$MU_DEV_XIDS})"
  ( cd "$TOOLS" && $PY prepare_muaalem.py \
      --tokens "$DATA/tokens.txt" --out "$DATA" --wavs "$WAVS" \
      --evalsets "$EVALSETS" \
      --target-hours "$MU_TRAIN_HOURS" --cap-hours "$MU_CAP_HOURS" \
      --dev-xids "$MU_DEV_XIDS" --match-ratio-min "$MU_MATCH_RATIO_MIN" )
  mark s4b_muaalem
fi

# ----------------------------- S4c: retasy (eval only) -----------------------
if ! done_p s4c_retasy; then
  log "S4c RetaSy real-user eval sets"
  ( cd "$TOOLS" && $PY prepare_retasy.py \
      --labels "$DATA/labels.json" --wavs "$WAVS/retasy" --evalsets "$EVALSETS" )
  mark s4c_retasy
fi

# ----------------------------- S4d: prep gate ---------------------------------
if ! done_p s4d_gate; then
  log "S4d prep gate"
  ( cd "$TOOLS" && $PY check_prep.py ) || {
    rm -f "$STATE/s4a_everyayah.done" "$STATE/s4b_muaalem.done"
    fail "prep gate failed — S4a/S4b markers cleared; re-paste to retry the unconsumed shards"
  }
  mark s4d_gate
fi

# ----------------------------- S5: noise/RIR corpora -------------------------
if ! done_p musan_extracted; then
  log "S5 downloading MUSAN"
  rm -rf "$BASE/musan"; mkdir -p "$BASE/musan"
  curl -fL --retry 3 -o "$BASE/musan/musan.tar.gz" "$MUSAN_URL"
  # --no-same-owner: network volumes forbid chown; without it tar exits 2 on
  # thousands of harmless ownership warnings and set -e kills the run
  tar -xzf "$BASE/musan/musan.tar.gz" -C "$BASE/musan" --no-same-owner
  rm -f "$BASE/musan/musan.tar.gz"
  rm -rf "$BASE/musan/musan/speech"   # never mixed (intelligible speech injects competing phonemes)
  mark musan_extracted
fi
if ! done_p rirs_extracted; then
  log "S5 downloading RIRS_NOISES"
  rm -rf "$BASE/rirs"; mkdir -p "$BASE/rirs"
  curl -fL --retry 3 -o "$BASE/rirs/rirs_noises.zip" "$RIRS_URL"
  unzip -q "$BASE/rirs/rirs_noises.zip" -d "$BASE/rirs"
  rm -f "$BASE/rirs/rirs_noises.zip"
  mark rirs_extracted
fi

# --------------------- S5a/S5b: per-source features + staged wav delete ------
if ! done_p s5a_ea_feats; then
  log "S5a everyayah features (speed x3 + RIR ${RIR_FRACTION} view)"
  ( cd "$TOOLS" && $PY make_features.py --data "$DATA" --feats "$FEATS" \
      --source ea --rirs-dir "$BASE/rirs/RIRS_NOISES" --rir-fraction "$RIR_FRACTION" )
  ( cd "$TOOLS" && $PY cleanup_train_wavs.py "$DATA/ea_cuts_train_raw.jsonl.gz" \
      "$EVALSETS/ea_indev.jsonl" "$EVALSETS/ea_unseen.jsonl" )
  mark s5a_ea_feats
fi
if ! done_p s5b_mu_feats; then
  log "S5b muaalem features (speed x3 + RIR ${RIR_FRACTION} view)"
  ( cd "$TOOLS" && $PY make_features.py --data "$DATA" --feats "$FEATS" \
      --source mu --rirs-dir "$BASE/rirs/RIRS_NOISES" --rir-fraction "$RIR_FRACTION" )
  ( cd "$TOOLS" && $PY cleanup_train_wavs.py "$DATA/mu_cuts_train_raw.jsonl.gz" \
      "$EVALSETS/mu_unseen.jsonl" )
  mark s5b_mu_feats
fi
if ! done_p s5c_merge; then
  log "S5c MUSAN/RIRS noise manifests + train manifest merge"
  ( cd "$TOOLS" && $PY make_features.py --data "$DATA" --feats "$FEATS" \
      --source musan --musan-dir "$BASE/musan/musan" --rirs-dir "$BASE/rirs/RIRS_NOISES" )
  ( cd "$TOOLS" && $PY make_features.py --data "$DATA" --feats "$FEATS" --source indev )
  ( cd "$TOOLS" && DATA="$DATA" $PY merge_train.py )
  mark s5c_merge
fi

# ----------------------------- S6: recipe assembly ---------------------------
if ! done_p s6_recipe; then
  log "S6 assembling recipe (stock zipformer @ ${ICEFALL_COMMIT:0:10} + 2 patches)"
  SRC="$BASE/icefall/egs/librispeech/ASR/zipformer"
  cp -fL "$SRC"/*.py "$RECIPE/"
  for f in train.py model.py zipformer.py export-onnx-streaming-ctc.py \
           decoder.py joiner.py encoder_interface.py; do
    [ -f "$RECIPE/$f" ] || fail "recipe file missing after copy: $f"
  done
  $PY - <<'PYEOF'
import os
recipe = os.path.join(os.environ["BASE"], "recipe", "train.py")
src = open(recipe, encoding="utf-8").read()
patches = [
    # our unit tokenizer instead of sentencepiece (same duck-typed surface)
    ("import sentencepiece as spm", "import unit_tokenizer as spm"),
    # prep caps at 26.5s; 0.9x speed stretches to 29.4s — keep up to 30.5s
    ("if c.duration < 1.0 or c.duration > 20.0:",
     "if c.duration < 1.0 or c.duration > 30.5:"),
]
for old, new in patches:
    assert src.count(old) == 1, f"patch anchor not found exactly once: {old!r}"
    src = src.replace(old, new)
open(recipe, "w", encoding="utf-8", newline="\n").write(src)
print("[recipe] train.py patched (tokenizer + 30.5s filter)")
PYEOF
  mark s6_recipe
fi
# tool copies refresh EVERY paste (edits must reach training; audit M13)
cp -f "$TOOLS/unit_tokenizer.py" "$RECIPE/"
cp -f "$TOOLS/asr_datamodule.py" "$RECIPE/"

train_common_args() {
  echo --world-size "$WORLD_SIZE" \
       --use-transducer 0 --use-ctc 1 \
       --causal 1 --chunk-size "$CHUNK_SIZES" \
       --use-bf16 1 \
       --bpe-model "$DATA/tokens.txt" \
       --manifest-dir "$DATA" \
       --max-duration "$MAX_DURATION" \
       --enable-musan 1 \
       --full-libri 1 \
       --save-every-n 100000 --keep-last-k 2
}

# ----------------------------- S7: smoke gate (real batch size) --------------
SMOKE_MAX_DURATION=$(( GLOBAL_BATCH_SEC / TRAIN_GPUS ))   # the S8 per-GPU batch, regardless of this pod's GPU count
if ! done_p s7_smoke; then
  log "S7 smoke run (8h subset, TRAIN-pod max_duration=$SMOKE_MAX_DURATION, 1 epoch, export, sherpa decode)"
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
    if total >= 8 * 3600:
        break
CutSet.from_cuts(sub).to_file(f"{base}/smoke/data/quran_cuts_train.jsonl.gz")
dev = CutSet.from_file(f"{base}/data/quran_cuts_dev.jsonl.gz")
sdev = [c for i, c in enumerate(dev) if i < 60]
assert sdev, "dev subset empty"
CutSet.from_cuts(sdev).to_file(f"{base}/smoke/data/quran_cuts_dev.jsonl.gz")
print(f"[smoke] subset: {total / 3600:.2f}h train, {len(sdev)} dev cuts")
PYEOF
  cp -f "$DATA/musan_cuts.jsonl.gz" "$SMOKE/data/" 2>/dev/null || true
  T0=$(date +%s)
  ( cd "$RECIPE" && $PY train.py $(train_common_args) \
      --manifest-dir "$SMOKE/data" \
      --world-size 1 --max-duration "$SMOKE_MAX_DURATION" --num-buckets 2 \
      --exp-dir "$SMOKE/exp" --num-epochs 1 --start-epoch 1 \
    2>&1 | tee "$SMOKE/train.log" )
  T1=$(date +%s)
  grep -q "Epoch 1" "$SMOKE/train.log" || fail "smoke train produced no epoch log"
  [ -f "$SMOKE/exp/epoch-1.pt" ] || fail "smoke train produced no checkpoint"
  # measured throughput -> projected wall/cost for the full run on the S8 pod
  FULL_H=$($PY -c "
import os
from lhotse import CutSet
cs = CutSet.from_file(os.environ['DATA'] + '/quran_cuts_train.jsonl.gz')
print(int(sum(c.duration for c in cs) / 3600))")
  awk -v dt="$((T1 - T0))" -v subh=8 -v full="$FULL_H" -v tg="$TRAIN_GPUS" -v ep="$NUM_EPOCHS" -v rate="$GPU_RATE" 'BEGIN{
    eph = dt / 3600 * full / subh / tg;
    printf "[smoke] measured: %.0fs for %dh on 1 GPU; full set %dh -> projected %.1fh/epoch on %d GPUs, %.1fh total, ~$%.0f\n", dt, subh, full, eph, tg, eph * ep, eph * ep * tg * rate
  }'

  ( cd "$RECIPE" && $PY export-onnx-streaming-ctc.py \
      --exp-dir "$SMOKE/exp" --tokens "$DATA/tokens.txt" \
      --epoch 1 --avg 1 --use-averaged-model 0 \
      --use-transducer 0 --use-ctc 1 --causal 1 \
      --chunk-size 8 --left-context-frames "$EXPORT_LEFT" \
    2>&1 | tee "$SMOKE/export.log" )
  SMOKE_ONNX="$SMOKE/exp/ctc-epoch-1-avg-1-chunk-8-left-${EXPORT_LEFT}.int8.onnx"
  [ -f "$SMOKE_ONNX" ] || SMOKE_ONNX=$(ls "$SMOKE"/exp/*.int8.onnx 2>/dev/null | head -1 || true)
  [ -n "$SMOKE_ONNX" ] && [ -f "$SMOKE_ONNX" ] || fail "smoke export produced no int8 onnx"

  $PY - "$SMOKE_ONNX" <<'PYEOF'
import os, sys
import sherpa_onnx
import soundfile as sf
from lhotse import CutSet
base = os.environ["BASE"]
model = sys.argv[1]
rec = sherpa_onnx.OnlineRecognizer.from_zipformer2_ctc(
    tokens=f"{base}/data/tokens.txt", model=model, num_threads=2)
dev = list(CutSet.from_file(f"{base}/smoke/data/quran_cuts_dev.jsonl.gz"))[:2]
assert dev, "no smoke dev cuts to decode"
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
  log "S7 smoke PASSED (1-epoch output is near-garbage by design; the full run fixes that)"
fi

# ----------------------------- S7b: DDP sanity (multi-GPU pods only) ---------
if [ "$WORLD_SIZE" -gt 1 ] && ! done_p s7b_ddp; then
  log "S7b DDP sanity: 1 epoch on an 8h subset at world-size $WORLD_SIZE"
  DDPS="$BASE/ddp_smoke"; rm -rf "$DDPS"; mkdir -p "$DDPS/data" "$DDPS/exp"
  $PY - <<'PYEOF'
import os
from lhotse import CutSet
base = os.environ["BASE"]
cuts = CutSet.from_file(f"{base}/data/quran_cuts_train.jsonl.gz")
sub, total = [], 0.0
for c in cuts:
    sub.append(c)
    total += c.duration
    if total >= 8 * 3600:
        break
CutSet.from_cuts(sub).to_file(f"{base}/ddp_smoke/data/quran_cuts_train.jsonl.gz")
dev = CutSet.from_file(f"{base}/data/quran_cuts_dev.jsonl.gz")
sdev = [c for i, c in enumerate(dev) if i < 60]
CutSet.from_cuts(sdev).to_file(f"{base}/ddp_smoke/data/quran_cuts_dev.jsonl.gz")
print("[ddp-smoke] subset written")
PYEOF
  cp -f "$DATA/musan_cuts.jsonl.gz" "$DDPS/data/" 2>/dev/null || true
  nvidia-smi topo -m || true
  ( cd "$RECIPE" && $PY train.py $(train_common_args) \
      --manifest-dir "$DDPS/data" --num-buckets 2 \
      --exp-dir "$DDPS/exp" --num-epochs 1 --start-epoch 1 \
    2>&1 | tee "$DDPS/train.log" )
  [ -f "$DDPS/exp/epoch-1.pt" ] || fail "DDP smoke produced no checkpoint — check NCCL/topology"
  rm -rf "$DDPS"
  mark s7b_ddp
  log "S7b DDP sanity PASSED"
fi

# ----------------------------- phase gate ------------------------------------
# The prep pod (fewer GPUs than TRAIN_GPUS) must not roll into the full train:
# it would run at TRAIN_GPUS x the wall-clock, or OOM at the merged batch size.
# Intentionally training on THIS pod anyway: set TRAIN_GPUS to its GPU count.
if ! done_p s8_train && [ "$WORLD_SIZE" -lt "$TRAIN_GPUS" ]; then
  log "PREP PHASE COMPLETE (this pod has $WORLD_SIZE GPU(s); training wants $TRAIN_GPUS)"
  echo "=============================================================="
  echo " Next:"
  echo " 1. Upload the SHIPPED model now (the eval needs it):"
  echo "      $BASE/incumbent/model.int8.onnx  <- app assets/models/quran-phoneme-160/model.int8.onnx"
  echo "      $BASE/incumbent/tokens.txt       <- app assets/models/quran-phoneme-160/tokens.tokens (renamed)"
  echo " 2. STOP this pod (it bills while idle)."
  echo " 3. Deploy ${TRAIN_GPUS}x A100 80GB SXM Secure on the SAME volume, then run:"
  echo "      EXPECTED_GPUS=$TRAIN_GPUS bash <(curl -fsSL https://raw.githubusercontent.com/Nour-benmohamed-Git/noor-data/master/train/run_quran_train_v2.sh)"
  echo "=============================================================="
  exit 0
fi

# ----------------------------- S8: full training -----------------------------
if ! done_p s8_train; then
  START=1
  LAST=$(ls "$EXP"/epoch-*.pt 2>/dev/null | sed 's/.*epoch-\([0-9]*\)\.pt/\1/' | sort -n | tail -1 || true)
  if [ -n "${LAST:-}" ]; then START=$((LAST + 1)); fi
  if [ "$START" -gt "$NUM_EPOCHS" ]; then
    log "S8 training already complete (epoch $LAST)"
  else
    log "S8 training epochs $START..$NUM_EPOCHS (resume-safe; re-paste to resume)"
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
        C=$(awk -v h="$H" -v g="$WORLD_SIZE" -v r="$GPU_RATE" 'BEGIN{printf "%.2f", h*r*g}')
        E=$(ls "$EXP"/epoch-*.pt 2>/dev/null | wc -l || true)
        echo "[quran-v2] elapsed ${H}h (~\$${C} at \$${GPU_RATE}/GPU-h) — epoch checkpoints: ${E:-0}/$NUM_EPOCHS"
        # retain the last 6 epoch checkpoints (resume + export avg window)
        ls "$EXP"/epoch-*.pt 2>/dev/null | sed 's/.*epoch-\([0-9]*\)\.pt/\1/' | sort -n | head -n -6 | \
          while read -r OLD; do rm -f "$EXP/epoch-$OLD.pt"; done
        # batch checkpoints: train.py keeps 2 (--keep-last-k); belt-and-braces prune to 1
        ls -t "$EXP"/checkpoint-*.pt 2>/dev/null | tail -n +2 | while read -r OLD; do rm -f "$OLD"; done
      done
    ) &
    MON_PID=$!
    TRAIN_RC=0
    wait $TRAIN_PID || TRAIN_RC=$?
    kill $MON_PID 2>/dev/null || true
    [ "$TRAIN_RC" -eq 0 ] || fail "training exited with $TRAIN_RC — re-paste to resume (CUDA OOM? resume with GLOBAL_BATCH_SEC=2400)"
  fi
  LASTCK=$(ls "$EXP"/epoch-*.pt 2>/dev/null | sed 's/.*epoch-\([0-9]*\)\.pt/\1/' | sort -n | tail -1 || true)
  [ -n "$LASTCK" ] && [ "$LASTCK" -ge "$NUM_EPOCHS" ] && mark s8_train || \
    fail "training finished but newest checkpoint is epoch-${LASTCK:-none} (< $NUM_EPOCHS)"
fi

# ----------------------------- S9: export (both latency profiles) ------------
if ! done_p s9_export; then
  EXPORT_EPOCH=$(ls "$EXP"/epoch-*.pt 2>/dev/null | sed 's/.*epoch-\([0-9]*\)\.pt/\1/' | sort -n | tail -1 || true)
  [ -n "$EXPORT_EPOCH" ] || fail "no epoch checkpoints found in $EXP"
  for CH in $EXPORT_CHUNKS; do
    log "S9 exporting chunk $CH (epoch $EXPORT_EPOCH avg $EXPORT_AVG left $EXPORT_LEFT)"
    ( cd "$RECIPE" && $PY export-onnx-streaming-ctc.py \
        --exp-dir "$EXP" --tokens "$DATA/tokens.txt" \
        --epoch "$EXPORT_EPOCH" --avg "$EXPORT_AVG" \
        --use-transducer 0 --use-ctc 1 --causal 1 \
        --chunk-size "$CH" --left-context-frames "$EXPORT_LEFT" \
      2>&1 | tee "$BASE/export_ch$CH.log" )
    STEM="ctc-epoch-$EXPORT_EPOCH-avg-$EXPORT_AVG-chunk-$CH-left-$EXPORT_LEFT"
    [ -f "$EXP/$STEM.onnx" ] || fail "expected export missing: $EXP/$STEM.onnx"
    [ -f "$EXP/$STEM.int8.onnx" ] || fail "expected int8 export missing: $EXP/$STEM.int8.onnx"
    cp -f "$EXP/$STEM.onnx" "$EXP/model.ch$CH.fp32.onnx"
    cp -f "$EXP/$STEM.int8.onnx" "$EXP/model.ch$CH.int8.onnx"
    SZ=$(stat -c %s "$EXP/model.ch$CH.int8.onnx")
    [ "$SZ" -le 78643200 ] || fail "int8 chunk-$CH is $((SZ / 1048576)) MiB (>75) — too big to bundle"
  done
  mark s9_export
fi

# ----------------------------- S10: eval matrix -------------------------------
if ! done_p s10_eval; then
  log "S10 eval matrix (incumbent + 4 candidates x 5 sets + probes)"
  if [ ! -f "$BASE/incumbent/model.int8.onnx" ] || [ ! -f "$BASE/incumbent/tokens.txt" ]; then
    fail "upload the SHIPPED model first:
  mkdir -p $BASE/incumbent
  # from your machine (app repo): assets/models/quran-phoneme-160/model.int8.onnx
  #                               assets/models/quran-phoneme-160/tokens.tokens -> tokens.txt
  # e.g. via the pod's Jupyter file browser or runpodctl send
then re-paste this script."
  fi
  INC_SHA=$(sha1sum "$BASE/incumbent/model.int8.onnx" | awk '{print $1}')
  if [ "$INC_SHA" != "$INCUMBENT_SHA1" ] && [ "${ALLOW_INCUMBENT_MISMATCH:-0}" != "1" ]; then
    fail "incumbent sha1 $INC_SHA != shipped $INCUMBENT_SHA1 (set ALLOW_INCUMBENT_MISMATCH=1 to override)"
  fi
  cmp -s "$BASE/incumbent/tokens.txt" "$DATA/tokens.txt" || \
    fail "incumbent tokens.txt differs from generated tokens.txt — wrong file uploaded"
  MODELS="shipped=$BASE/incumbent/model.int8.onnx"
  MODELS="$MODELS,new_fp32_ch8=$EXP/model.ch8.fp32.onnx,new_int8_ch8=$EXP/model.ch8.int8.onnx"
  [ -f "$EXP/model.ch16.int8.onnx" ] && MODELS="$MODELS,new_int8_ch16=$EXP/model.ch16.int8.onnx"
  ( cd "$TOOLS" && $PY eval_matrix.py \
      --models "$MODELS" --tokens "$DATA/tokens.txt" \
      --evalsets "$EVALSETS" \
      --eval-noises "$DATA/eval_noises.txt" --eval-rirs "$DATA/eval_rirs.txt" \
      --limit "$EVAL_LIMIT" --report "$BASE/eval_matrix.json" )
  mark s10_eval
fi

# ----------------------------- S11: ship gates -> $OUT ------------------------
if ! done_p s11_gate; then
  log "S11 ship gates"
  ( cd "$TOOLS" && $PY gate.py --matrix "$BASE/eval_matrix.json" \
      --report "$BASE/gates_report.json" ) || {
    cp -f "$BASE/eval_matrix.json" "$BASE/gates_report.json" "$OUT/" 2>/dev/null || true
    fail "SHIP GATES FAILED — reports copied to $OUT for inspection; artifacts NOT released"
  }
  for CH in $EXPORT_CHUNKS; do
    cp -f "$EXP/model.ch$CH.int8.onnx" "$OUT/"
    cp -f "$EXP/model.ch$CH.fp32.onnx" "$OUT/" 2>/dev/null || true
  done
  cp -f "$DATA/tokens.txt" "$OUT/tokens.txt"
  cp -f "$BASE/eval_matrix.json" "$BASE/gates_report.json" "$OUT/"
  cp -f "$DATA/ea_coverage.json" "$OUT/" 2>/dev/null || true
  ( cd "$OUT" && sha1sum model.*.onnx tokens.txt | tee SHA1SUMS )
  ls -la "$OUT"
  mark s11_gate
fi

log "ALL DONE."
echo "=============================================================="
echo " Artifacts in $OUT :"
echo "   model.ch8.int8.onnx (160ms)   model.ch16.int8.onnx (320ms)"
echo "   tokens.txt  eval_matrix.json  gates_report.json  SHA1SUMS"
echo
echo " Next (on your machine):"
echo "   1. download \$OUT"
echo "   2. app swap: assets/models/quran-phoneme-160-v2/, AsrModel.ts"
echo "      (MODEL_DIR 'noor-phoneme-v2', expectedBytes, stale-dir list)"
echo "   3. npm run test:phoneme-matcher -- --full"
echo "   4. device side-by-side: old vs new, chunk8 vs chunk16"
echo " Then STOP THE POD."
echo "=============================================================="
