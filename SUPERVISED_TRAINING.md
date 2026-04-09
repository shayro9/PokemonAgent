# 🎓 Supervised Pretraining — From Raw Replays to a Warm-Started Policy

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your-username/PokemonAgent/blob/main/PokemonAgent_Colab_Training.ipynb)

Bootstraps the `AttentionPointerPolicy` from human Pokémon Showdown replays before
any RL training begins.  The pretraining step runs **behavioral cloning** (match human
moves) and **value pretraining** (predict game outcome) jointly, giving the RL agent a
useful starting point instead of random play.

```
Raw replays (HuggingFace)
       │
       ▼  download_metamon_hf.py
 .tar.gz  →  extracted .json.lz4 files
       │
       ▼  convert_metamon_json.py
 gen1ou.jsonl  (1 timestep per line, obs = BattleStateGen1.to_array())
       │
       ▼  run_supervised.py
 supervised_gen1_6v6.zip  (pretrained MaskablePPO checkpoint)
       │
       ▼  training/train.py  --load-model-path ...
 RL fine-tuning continues from warm weights
```

---

## Table of Contents

1. [Google Colab (Quickstart)](#1-google-colab-quickstart)
2. [Prerequisites](#2-prerequisites)
3. [Step 1 — Download replay data](#3-step-1--download-replay-data)
4. [Step 2 — Convert replays to observation arrays](#4-step-2--convert-replays-to-observation-arrays)
5. [Step 3 — Run supervised pretraining](#5-step-3--run-supervised-pretraining)
6. [Step 4 — Continue with RL fine-tuning](#6-step-4--continue-with-rl-fine-tuning)
7. [Alternative — Inline warmup inside the RL pipeline](#7-alternative--inline-warmup-inside-the-rl-pipeline)
8. [Observation format](#8-observation-format)
9. [Hyperparameter reference](#9-hyperparameter-reference)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Google Colab (Quickstart)

The fastest way to run supervised pretraining is directly in the provided Colab notebook.

**[📓 Open PokemonAgent_Colab_Training.ipynb](https://colab.research.google.com/github/your-username/PokemonAgent/blob/main/PokemonAgent_Colab_Training.ipynb)**

The notebook walks through the full pipeline in runnable cells:

| Cell | What it does |
|------|--------------|
| **S1** | Install `huggingface_hub` + `lz4` |
| **S2** | Download Gen 1 OU replays from HuggingFace |
| **S3** | Convert `.json.lz4` replays → `gen1ou.jsonl` |
| **S4** | Run supervised pretraining → `models/supervised_gen1_6v6.zip` |
| **7**  | RL fine-tuning from the checkpoint (or from scratch) |

> **Tip:** Run Cells S1–S4 before Cell 7 to start RL from a warm, human-like policy.  
> Skip S1–S4 to train from a random policy instead.

---

## 2. Prerequisites

Install Python dependencies (if you haven't already):

```bash
pip install -r requirements.txt
pip install huggingface_hub lz4
```

> `huggingface_hub` is only needed for the download step.  
> `lz4` is needed for the conversion step — the replay files are lz4-compressed.

---

## 3. Step 1 — Download replay data

Replay data comes from the
[jakegrigsby/metamon-parsed-replays](https://huggingface.co/datasets/jakegrigsby/metamon-parsed-replays)
HuggingFace dataset.  Each format is a `.tar.gz` of pre-parsed `.json.lz4` battle files.

```bash
# Download Gen 1 OU replays (~few hundred MB)
python data_generation/download_metamon_hf.py \
    --format gen1ou \
    --output-dir data/metamon

# Download multiple tiers at once
python data_generation/download_metamon_hf.py \
    --format gen1ou gen1uu gen1ubers \
    --output-dir data/metamon

# See all available formats
python data_generation/download_metamon_hf.py --list
```

After this step you will have:

```
data/
  metamon/
    gen1ou/
      *.json.lz4      ← one file per battle
```

### Available Gen 1 formats

| Format | Tier |
|--------|------|
| `gen1ou` | OverUsed (most competitive) |
| `gen1uu` | UnderUsed |
| `gen1nu` | NeverUsed |
| `gen1ubers` | Ubers |

> **Note:** Only Gen 1 formats are supported by the current observation pipeline.
> Gen 2–9 formats exist on HuggingFace but require a separate state class.

---

## 4. Step 2 — Convert replays to observation arrays

This step reads every battle file, runs each timestep through
`BattleStateGen1.to_array()` via a proxy adapter, and writes one JSON record
per timestep to a `.jsonl` file.

```bash
python data_generation/convert_metamon_json.py \
    --input-dir data/metamon/gen1ou \
    --output    data/supervised/gen1ou.jsonl
```

### Common options

| Flag | Default | Description |
|------|---------|-------------|
| `--input-dir` | *(required)* | Directory containing `.json.lz4` battle files |
| `--output` | *(required)* | Output `.jsonl` path |
| `--max-battles` | all | Process only the first N battles (useful for quick tests) |
| `--obs-version` | `6v6` | Always use `6v6` — `v1-deprecated` is removed |

### Quick test (100 battles)

```bash
python data_generation/convert_metamon_json.py \
    --input-dir data/metamon/gen1ou \
    --output    data/supervised/gen1ou_test.jsonl \
    --max-battles 100
```

### What the converter produces

Each line in the `.jsonl` file is one battle timestep:

```json
{
  "obs":         [0.033, 0.183, ...],  // 1279 floats — BattleStateGen1.to_array()
  "action":      8,                    // int in [0,9]: 0-5 = switch slot, 6-9 = move slot
  "action_type": "move",               // "move" | "switch"
  "battle_id":   "gen1ou-12345",       // groups steps into an episode
  "turn":        3,                    // zero-based turn index
  "player":      "p1",
  // only on the last step of each battle:
  "outcome":     1                     // +1 win, -1 loss
}
```

### Observation size

The observation vector is **1279 floats**, structured identically to what the
live environment produces:

```
arena state           5   (turn + Reflect/Light Screen for both sides)
opp active moves    140   (4 moves × 35-dim MoveState)
opp bench           156   (6 slots × 25 dims + 6-dim alive vector)
my bench            978   (6 slots × 162 dims + 6-dim alive vector)
─────────────────  ────
total              1279
```

> My bench slots each contain the full `MyPokemonStateGen1` (HP, stats, boosts,
> status, effects, STAB) **plus** that Pokémon's 4 moves, so the policy sees
> its entire team at once.

---

## 5. Step 3 — Run supervised pretraining

```bash
python -m training.supervised.run_supervised \
    --data-path  data/supervised/gen1ou.jsonl \
    --epochs     20 \
    --val-split  0.1 \
    --output-path models/supervised_gen1_6v6.zip
```

Training prints one line per epoch:

```
[pretrain] Loaded 48 312 episodes / 512 088 steps (skipped 1 024 invalid)
[pretrain] Train: 43 480 episodes | Val: 4 832 episodes (51 455 steps)
[pretrain] 460 633 train steps | epochs=20 | λ_v=1.0 λ_bc=1.0
  epoch   1/20 | total=2.3184  value=1.2041  bc=1.1143  val_acc=31.4%
  epoch   2/20 | total=2.1902  value=1.1263  bc=1.0639  val_acc=35.8%
  ...
  epoch  20/20 | total=1.7441  value=0.9812  bc=0.7629  val_acc=47.3%
[pretrain] Done.
[run_supervised] Model saved to models/supervised_gen1_6v6.zip
```

### What the trainer is optimising

```
L = λ_value · MSE(V(s), Gₜ)  +  λ_bc · CrossEntropy(π(s), a_human)
```

- **Value head** is trained to predict discounted Monte-Carlo returns
  (win → +15, loss → −10 at the terminal step, then discounted back at γ=0.99).
- **Policy head** is trained to imitate the human's move choice via cross-entropy.
- All 10 action logits (0–9) are computed; if an `action_mask` is present in
  the data, illegal actions are masked to −∞ before the CE loss so the policy
  is never penalised for not choosing unavailable moves.

### Key flags

| Flag | Default | Description |
|------|---------|-------------|
| `--data-path` | *(required)* | `.jsonl` file from Step 2 |
| `--epochs` | `10` | Number of passes through the dataset |
| `--lr` | `3e-4` | Adam learning rate |
| `--batch-size` | `256` | Mini-batch size |
| `--val-split` | `0.0` | Fraction of episodes held out for accuracy tracking |
| `--lambda-value` | `1.0` | Weight of the value MSE term |
| `--lambda-bc` | `1.0` | Weight of the behavioral cloning CE term |
| `--device` | `auto` | `cpu`, `cuda`, or `auto` |
| `--output-path` | `models/supervised_gen1_6v6.zip` | Where to save the checkpoint |

---

## 6. Step 4 — Continue with RL fine-tuning

The saved `.zip` is a standard `MaskablePPO` checkpoint that can be loaded
directly into the RL training pipeline:

```bash
python -m training.train \
    --load-model-path models/supervised_gen1_6v6 \
    --timesteps 500000
```

> The `--load-model-path` flag accepts the path **with or without** the `.zip`
> extension.

To evaluate the supervised checkpoint before any RL training (useful to
measure the BC baseline win rate):

```bash
python -m training.train \
    --load-model-path models/supervised_gen1_6v6 \
    --timesteps 0 \
    --eval-after-warmup \
    --eval-episodes 200
```

---

## 7. Alternative — Inline warmup inside the RL pipeline

If you want supervised pretraining to happen automatically at the start of
an RL run (no separate Step 3), pass `--warmup-data-path` to `train.py`:

```bash
python -m training.train \
    --warmup-data-path data/supervised/gen1ou.jsonl \
    --warmup-epochs    20 \
    --timesteps        1000000
```

This is equivalent to running Steps 3 and 4 together.  The pipeline is:

```
warmup (BC + value) → [optional baseline eval] → RL training (MaskablePPO)
```

### From Python

```python
from training.supervised import SupervisedWarmupConfig
from training.train import train_model

train_model(
    ...,
    warmup_config=SupervisedWarmupConfig(
        data_path="data/supervised/gen1ou.jsonl",
        n_epochs=20,
        val_split=0.1,
    ),
)
```

---

## 8. Observation format

The pipeline reuses `BattleStateGen1` — the same class the live environment
uses — so training data and RL rollouts are guaranteed to have identical
observation layouts.

The conversion path:

```
metamon state dict
  └── MetamonBattleProxy          adapts dict → poke-env-like interface
       └── BattleStateGen1        same class used in the live env
            └── .to_array()       → float32 (1279,)
```

**Opponent bench limitation:** Metamon replays only record the *active*
opponent Pokémon at each step.  Opponent bench slots 1–5 are zero-padded,
which mirrors the real game situation where you haven't yet seen those
Pokémon.

---

## 9. Hyperparameter reference

### Supervised warmup defaults (in `SupervisedWarmupConfig`)

| Parameter | Default | Notes |
|-----------|---------|-------|
| `n_epochs` | `10` | More epochs → higher BC accuracy, but risk of over-imitation |
| `batch_size` | `256` | Larger batches are stable; limited by GPU memory |
| `lr` | `3e-4` | Adam; same default as the RL run |
| `lambda_value` | `1.0` | Reduce if value loss dominates early |
| `lambda_bc` | `1.0` | Increase to prioritise action imitation |
| `gamma` | `0.99` | Discount for return computation |
| `val_split` | `0.0` | Set to `0.1` to monitor per-epoch accuracy |

### Architecture (shared with RL — defined in `training/config.py`)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `context_hidden` | `256` | Arena + opponent context encoder width |
| `move_hidden` | `128` | Per-move encoder width |
| `trunk_hidden` | `256` | Shared trunk before pointer heads |
| `n_attention_heads` | `4` | Multi-head attention in the team encoder |

> **Important:** These values must stay the same between supervised pretraining
> and RL fine-tuning.  Both pipelines read from `training/config.py::POLICY_KWARGS`
> so they are always in sync.

---

## 10. Troubleshooting

### `No JSON files found in ...`
You need to run the download step first, or point `--input-dir` at the
extracted directory (e.g. `data/metamon/gen1ou`, not `data/metamon`).

### `[pretrain] Loaded 0 episodes`
The `.jsonl` file exists but no records passed validation.  Check that:
- The file was produced with `--obs-version 6v6` (the default).
- The `obs` field has exactly 1279 values.
- The `action` field is an integer in `[0, 9]`.

### Validation accuracy plateaus below ~35%
Gen 1 OU has 10 valid actions per turn and human play is highly diverse, so
35–50% top-1 accuracy is normal and sufficient to warm-start RL.  The goal
is not to fully imitate humans, just to bootstrap away from random policy.

### CUDA out of memory
Reduce `--batch-size` (try 64 or 128).  The model is small so even CPU
training is feasible for a few thousand battles.

### Supervised checkpoint won't load into `train.py`
This happens if the architecture kwargs differ between training runs.
Both `run_supervised.py` and `train.py` read `POLICY_KWARGS` from
`training/config.py`, so they stay in sync automatically — but if you
manually overrode hidden sizes in one place you need to match them in the
other.
