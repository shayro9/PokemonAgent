# 🎮 PokemonAgent — Reinforcement Learning for Pokémon Showdown

A deep reinforcement learning agent that learns to play Pokémon 1v1 battles using **MaskablePPO** with a custom **Attention-Pointer Policy**. The agent builds a probabilistic model of the opponent's stats turn-by-turn and uses that belief to make informed move decisions.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Setup](#setup)
- [Generating a Training Dataset](#generating-a-training-dataset)
- [Training](#training)
- [Evaluation](#evaluation)
- [Key Concepts](#key-concepts)

---

## Overview

PokemonAgent connects to a local [Pokémon Showdown](https://github.com/smogon/pokemon-showdown) server via [poke-env](https://github.com/hsahovic/poke-env) and trains a neural network agent to win 1v1 Pokémon battles. The agent:

- **Observes** a rich 383-dimensional battle state encoding HP, stats, boosts, status conditions, move embeddings, and Bayesian opponent stat beliefs.
- **Acts** over 26 discrete actions (4 moves × variants + switches), with invalid actions masked out.
- **Learns** via MaskablePPO with a reward shaped around HP damage dealt/received, status effects, and win/loss outcomes.

---

## Architecture

### Attention-Pointer Policy

The core policy (`policy/attention_policy.py`) uses a permutation-equivariant architecture:

```
Flat Observation (383-dim)
         │
   ┌─────┴────────────┐
   │                  │
Context Vector    4 Move Embeddings
(everything        (MOVE_EMBED_LEN
 except moves)      each)
   │                  │
Context Encoder   Move Encoder
   │                  │
   └──→ Cross-Attention ←──┘
              │
         Trunk MLP
         /         \
  Pointer Head    Value Head
  (dot product     (scalar)
   per move)
```

- **Context encoder** — MLP over all non-move features
- **Move encoder** — Shared-weight MLP applied independently to each of the 4 move slots (permutation-equivariant)
- **Cross-attention** — Context attends over move embeddings to form an order-invariant summary
- **Pointer head** — Move logits computed as `dot(trunk_features, move_hidden_i)`, making move selection invariant to slot ordering

### Bayesian Stat Belief (`combat/stats_belief.py`)

Each turn, the agent maintains a **Gaussian posterior** over the opponent's 6 in-battle stats (HP, Atk, Def, SpA, SpD, Spe). The belief is updated from:

- **Damage dealt** → infers opponent Def or SpD
- **Damage received** → infers opponent Atk or SpA
- **Turn order** → infers opponent Speed

The resulting 12-dim belief vector `[mean×6, std×6]` is included in the observation, giving the agent explicit uncertainty quantification about the opponent.

### Protect Belief (`combat/protect_belief.py`)

A Bayesian model tracking the probability that the opponent will use a Protect move next turn, accounting for move accuracy and the diminishing-returns Protect mechanic.

---

## Project Structure

```
PokemonAgent/
├── agents/                  # Debug and utility agents
├── combat/                  # Battle math and belief systems
│   ├── combat_utils.py      # Damage modifiers, type chart helpers
│   ├── damage_estimate.py   # Per-move expected damage fraction
│   ├── event_parser.py      # Parses Showdown protocol events
│   ├── protect_belief.py    # Bayesian Protect model
│   ├── stat_belief_updates.py  # Wrappers to update StatBelief from battle
│   └── stats_belief.py      # Gaussian posterior over opponent stats
├── config/
│   └── config.py            # CLI argument resolution and opponent config
├── debug/                   # Debug utilities and log helpers
├── env/                     # Gymnasium environment wrappers
│   ├── action_masking.py    # Valid action mask builder
│   ├── battle_state.py      # 383-dim observation builder (BattleState)
│   ├── battle_tracker.py    # Per-battle mutable state (HP, move history)
│   ├── embed.py             # Move/status/weather feature encoders
│   ├── env_builder.py       # Environment factory
│   ├── reward.py            # Reward function
│   └── singles_env_wrapper.py  # Main SinglesEnv subclass
├── policy/
│   └── attention_policy.py  # AttentionPointerPolicy (MaskablePPO)
├── scripts/
│   ├── generate_db.js       # Node.js dataset generator
│   └── HelloWorldAgent.py   # Simple challenge script
├── teams/
│   ├── single_teams.py      # Predefined team strings
│   └── team_generators.py   # Random team generators from dataset
├── training/
│   ├── battle_metrics_log.py  # W&B battle metrics callback
│   ├── evaluation.py          # Model evaluation utilities
│   ├── parse.py               # CLI argument parser
│   └── train.py               # Main training entry point
├── replays/                 # Saved battle replays
├── data/                    # Generated datasets (gitignored)
└── requirements.txt
```

---

## Requirements

- Python 3.12+
- Node.js (for dataset generation via Pokémon Showdown sim)
- A running local [Pokémon Showdown server](https://github.com/smogon/pokemon-showdown)

---

## Setup

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd PokemonAgent
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up Pokémon Showdown (Local Server)

```bash
git clone https://github.com/smogon/pokemon-showdown.git
cd pokemon-showdown
node build
node pokemon-showdown start --no-security
```

The server runs at `localhost:8000` by default.

### 4. Install Node.js Dependencies (for dataset generation)

```bash
cd pokemon-showdown
npm install
cd ..
```

---

## Generating a Training Dataset

The agent can train against randomly generated Pokémon from the Gen 9 Random Battle format. First generate a dataset:

```bash
```

This produces `data/matchups_gen9randombattle_db.json` (or `data/gen9randombattle_db.json` depending on `MODE`). You can configure the script at the top:

| Variable | Description | Default |
|---|---|---|
| `FORMAT` | Showdown format to generate from | `gen9randombattle` |
| `MODE` | `'teams'` (one mon per entry) or `'matchups'` (paired 1v1) | `matchups` |
| `NUM_TO_GENERATE` | Number of matchups/teams to collect | `10000` |
| `DEDUPE_EXACT_SETS` | Skip duplicate sets | `true` |

---

## Training

### Quick Start — Predefined Teams

Train against all built-in single-Pokémon teams:

```bash
python -m training.train \
  --train-team steelix \
  --pool-all \
  --timesteps 100000 \
  --rounds-per-opponent 2000
```

### Train Against a Specific Opponent

```bash
python -m training.train \
  --train-team garchomp \
  --pool toxapex,corviknight,hippowdon \
  --timesteps 50000
```

### Train with Generated Random Teams

```bash
python -m training.train \
  --random-generated \
  --matchup-data-path data/matchups_gen9randombattle_db.json \
  --timesteps 200000 \
  --rounds-per-opponent 1
```

### Train with Train/Eval Split

```bash
python -m training.train \
  --random-generated \
  --matchup-data-path data/matchups_gen9randombattle_db.json \
  --split-generated-pool \
  --train-split 0.8 \
  --timesteps 200000 \
  --eval-every-timesteps 20000 \
  --eval-episodes 50
```

### All Training Arguments

| Argument | Description | Default |
|---|---|---|
| `--format` | Showdown battle format | `gen9customgame` |
| `--seed` | Global random seed | `42` |
| `--model-path` | Where to save the trained model | `data/1v1` |
| `--train-team` | Predefined agent team name | `None` (generated) |
| `--timesteps` | Total training timesteps | `20000` |
| `--rounds-per-opponent` | Battles before rotating opponent | `2000` |
| `--pool` | Comma-separated opponent names | — |
| `--pool-all` | Use all predefined solo teams | `False` |
| `--random-generated` | Use generated opponents | `False` |
| `--matchup-data-path` | Path to paired matchup dataset | — |
| `--agent-data-path` | Dataset for agent team generator | — |
| `--opponent-data-path` | Dataset for opponent team generator | — |
| `--split-generated-pool` | Split dataset into train/eval | `False` |
| `--train-split` | Fraction for training split | `0.8` |
| `--eval-every-timesteps` | Periodic evaluation interval | `0` |
| `--eval-episodes` | Episodes per evaluation | `0` |
| `--skip-eval` | Skip final evaluation | `False` |

### Available Predefined Teams

`steelix`, `garchomp`, `conkeldurr`, `rotom_wash`, `corviknight`, `toxapex`, `excadrill`, `hippowdon`, `breloom`, `volcarona`, `regigigas`, `bellibolt`

---

## Evaluation

Evaluation runs automatically after training (unless `--skip-eval` is passed). To run standalone evaluation after training:

```bash
python -m training.train \
  --model-path data/1v1 \
  --timesteps 0 \
  --eval-episodes 100 \
  --eval-pool-all \
  --skip-eval false
```

Metrics are also logged to **Weights & Biases** automatically. Set your W&B API key:

```bash
export WANDB_API_KEY=<your_key>
```

---

## Key Concepts

### Observation Space (383 dimensions)

| Feature Group | Dims | Description |
|---|---|---|
| Turn number | 1 | Normalized turn counter |
| Weather | 9 | One-hot weather encoding |
| My HP | 1 | Current HP fraction |
| My stats | 6 | Normalized stat values |
| My boosts | 7 | Stat boost stages |
| My status | 7 | Status condition one-hot |
| My effects | 3 | Active effects (Confusion, etc.) |
| Opp HP | 1 | Opponent HP fraction |
| Opp stat belief | 12 | Posterior mean+std for 6 stats |
| Opp boosts | 7 | Opponent stat boost stages |
| Opp status/effects | 11 | Opponent conditions |
| My moves | 4 × 39 | Embedded move features |
| Opp moves | 4 × 39 | Opponent revealed moves |
| Protect belief | 1 | P(opponent uses Protect) |

### Reward Function

| Event | Reward |
|---|---|
| Dealing damage | `+HP_fraction_dealt` |
| Taking damage | `-HP_fraction_taken` |
| Inflicting status on opponent | `+0.2 to +0.8` (status-dependent) |
| Receiving status | `-0.2 to -0.8` |
| Winning the battle | `+15.0` |
| Losing the battle | `-10.0` |

### Action Space (26 actions)

| Range | Action |
|---|---|
| 0–5 | Switch to party slot 0–5 |
| 6–9 | Use move slot 0–3 |
| 10–13 | Mega-evolve + move 0–3 |
| 14–17 | Z-move 0–3 |
| 18–21 | Dynamax + move 0–3 |
| 22–25 | Terastallize + move 0–3 |

Invalid actions are masked to zero probability before sampling.
