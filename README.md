<img width="960" height="540" alt="Screenshot_743" src="https://github.com/user-attachments/assets/92794399-6618-4485-a791-503845959ef8" />

# PokemonAgent

> Train and evaluate reinforcement-learning agents for local Pokemon Showdown battles, then challenge them in the browser.

PokemonAgent combines a local Showdown server, `poke-env` environment wrappers, dataset-driven team sampling, and a custom `MaskablePPO` policy. The current default battle configuration is Gen 1, and the repository ships with bundled Gen 1 matchup datasets plus a browser-play script and a Colab notebook.

## Why this repo exists

If you want a tight local loop for Pokemon RL, you usually have to glue together Showdown, battle-state encoding, action masking, training code, and evaluation yourself. This repo gives you that loop in one place, with a policy architecture that scores both moves and switches and a curriculum system that can swap opponent controllers over training time.

## What you can do

- Train a policy with `python -m training.train`
- Run periodic evaluation during training
- Switch opponent `Player` implementations with a YAML curriculum
- Challenge a trained, random, heuristic, or max-power bot with `python -m scripts.play_policy`
- Use the included Colab notebook for GPU training

## Quick start

1. Install Python dependencies.

```bash
pip install -r requirements.txt
```

2. Install and start the local Showdown server from the vendored `pokemon-showdown` directory.

```bash
cd pokemon-showdown
npm install
node build
node pokemon-showdown start --no-security
```

3. In a second terminal, run a short Gen 1 training job.

```bash
python -m training.train \
  --format gen1ou \
  --matchup-data-path data/matchups_1v1_gen1ou_db.json \
  --timesteps 10000 \
  --eval-episodes 20 \
  --model-path models/quickstart_gen1
```

This saves `models/quickstart_gen1.zip`.

> **Note**
> The bundled datasets in `data/` are Gen 1 OU files. The examples below pass `--format gen1ou` and explicit dataset paths on purpose.

## Prerequisites

- Python 3.12
- Node.js
- A local Pokemon Showdown checkout in `pokemon-showdown/` (already present in this repo)

## Setup

Clone your fork or this repository, then install the Python and Node dependencies shown above.

If you want a clean first run, keep two terminals open:

1. Terminal 1 runs `node pokemon-showdown start --no-security`
2. Terminal 2 runs training or `scripts.play_policy`

The local Showdown server listens on `http://localhost:8000`.

## Training

### Train from paired matchup data

Use `--matchup-data-path` when each JSON row already contains both the agent team and the opponent team.

```bash
python -m training.train \
  --format gen1ou \
  --matchup-data-path data/matchups_6v6_gen1ou_db.json \
  --split-generated-pool \
  --train-split 0.8 \
  --timesteps 100000 \
  --eval-every-timesteps 20000 \
  --eval-episodes 50 \
  --device auto \
  --n-envs 1 \
  --model-path models/gen1_6v6
```

### Train with an opponent curriculum

Use `--curriculum-config` to swap the opponent `Player` implementation at fixed global timesteps. Team generation stays the same; only the controller changes.

```bash
python -m training.train \
  --format gen1ou \
  --matchup-data-path data/matchups_6v6_gen1ou_db.json \
  --curriculum-config curriculum/examples/opponent_player_curriculum.yaml \
  --timesteps 50000 \
  --eval-episodes 25 \
  --model-path models/curriculum_gen1
```

The bundled example curriculum looks like this:

```yaml
stages:
  - name: warmup-random
    duration_timesteps: 500000
    opponent_player:
      id: random

  - name: midgame-max-power
    opponent_player:
      id: max-power
```

Each stage needs:

- `name`
- `duration_timesteps` or `end_timestep` for every non-final stage
- `opponent_player.id` or `opponent_player.class_path`
- optional `opponent_player.kwargs`

### Useful training flags

| Flag | What it does |
| --- | --- |
| `--format` | Sets the Showdown battle format |
| `--matchup-data-path` | Uses a paired matchup JSON file |
| `--agent-data-path` | Uses a team pool for the agent side |
| `--opponent-data-path` | Uses a team pool for the opponent side |
| `--split-generated-pool` | Splits generated data into disjoint train and eval pools |
| `--eval-every-timesteps` | Runs periodic evaluation during training |
| `--eval-episodes` | Sets the number of evaluation episodes |
| `--curriculum-config` | Loads a YAML opponent curriculum |
| `--device` | Chooses `auto`, `cuda`, or `cpu` |
| `--n-envs` | Runs multiple `SubprocVecEnv` workers |

For the full CLI, run:

```bash
python -m training.train --help
```

## Evaluation and logging

Final evaluation runs automatically unless you pass `--skip-eval`. If you want checkpoint-style evaluation during training, add `--eval-every-timesteps`.

Training also logs to Weights & Biases. Set your API key before you start if you want remote metrics:

```bash
export WANDB_API_KEY=<your_key>
```

PowerShell:

```powershell
$env:WANDB_API_KEY = "<your_key>"
```

## Play against a bot in the browser

Keep the local Showdown server running, open `http://localhost:8000`, and log in with the username you want the bot to challenge.

Then run:

```bash
python -m scripts.play_policy \
  --agent policy \
  --model-path models/quickstart_gen1 \
  --data-path data/matchups_1v1_gen1ou_db.json \
  --challenge-user MyName
```

`--agent` supports `policy`, `random`, `max-power`, and `heuristic`.

For the full browser-play walkthrough, see [PLAY_VS_AI.md](PLAY_VS_AI.md).

## Included docs

| File | Purpose |
| --- | --- |
| [PLAY_VS_AI.md](PLAY_VS_AI.md) | Step-by-step local browser battle flow |
| [COLAB_QUICK_START.md](COLAB_QUICK_START.md) | Fast Colab setup |
| [PokemonAgent_Colab_Training.ipynb](PokemonAgent_Colab_Training.ipynb) | Notebook for hosted training |

## Architecture

### Current default battle config

| Component | Value |
| --- | --- |
| Battle config | `BattleConfig.gen1()` |
| Observation size | `1279` floats |
| Action space | `10` actions (`6` switches + `4` moves) |
| RL algorithm | `sb3_contrib.MaskablePPO` |
| Policy class | `policy.policy.AttentionPointerPolicy` |

### Policy overview

The current policy uses separate encoders for battle context, active moves, and the full team:

- a context MLP for the global state
- a shared-weight move encoder for the 4 active move slots
- a shared-weight team encoder for the 6 team slots
- cross-attention from context into moves and team embeddings
- a move pointer head, a switch pointer head, and a value head

That design keeps move scoring and switch scoring slot-order-equivariant while still letting the policy condition on the whole battle state.

Action masking is provided by `PokemonRLWrapper.action_masks()`, so invalid actions are removed before sampling.

## Bundled datasets

The repo currently includes these Gen 1 JSON files in `data/`:

- `data/matchups_1v1_gen1ou_db.json`
- `data/matchups_6v6_gen1ou_db.json`
- `data/meta_pool_teams_6_gen1ou_db.json`

The training examples use explicit dataset paths so you can choose the exact source data instead of relying on CLI defaults.

## Project layout

```text
PokemonAgent/
├── agents/              # Playable bot wrappers, including the saved-policy player
├── combat/              # Battle math and combat helpers
├── config/              # CLI-side data and opponent resolution
├── curriculum/          # YAML curriculum models, loader, and runtime
├── data/                # Bundled matchup and team datasets
├── env/                 # Showdown environment wrappers and battle-state encoders
├── policy/              # Attention-pointer extractor and policy
├── pokemon-showdown/    # Local Showdown server checkout
├── scripts/             # Human-vs-bot entry points
├── teams/               # Team and matchup generators
├── tests/               # Unit tests
└── training/            # Training loop, evaluation, callbacks, and CLI parsing
```

## Testing

Run the test suite from the repo root:

```bash
pytest tests/ -q
```
