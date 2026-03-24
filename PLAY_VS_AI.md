# 🎮 Play Against an AI

Challenge any of the available AI agents in a real Pokémon Showdown browser battle — 3 steps.

---

## Step 1 — Start the local Showdown server

> ⚠️ **The server must be running before you do anything else.**  
> If you skip this step you will get a `Connect call failed` error.

Open a terminal in the repo root and run:

```bash
cd pokemon-showdown && node pokemon-showdown start --no-security
```

Leave this terminal running. You should see:
```
Workers: 1
Listening on 0.0.0.0:8000
```

---

## Step 2 — Build your team in the browser

> ⚠️ **You must do this before accepting any challenge, or you will get:**  
> *"You need to go into the Teambuilder and build a team for this format."*

1. Go to **http://localhost:8000** and click **Log in** — pick any username (e.g. `MyName`)
2. Click **Teambuilder** → **New Team**
3. Set the format to **Gen 1 OU** (or whatever format you are playing)
4. Add your Pokémon and save the team

> 💡 The script prints a suggested team for you under **"Your Team"** when it starts.  
> You can copy those Pokémon into the Teambuilder if you want to use them.

---

## Step 3 — Run the AI and send the challenge

Open a new terminal in the repo root. By default the AI's team is **randomly sampled** from the gen1ou dataset.

**Trained policy** (your saved model):
```bash
.venv/bin/python -m scripts.play_policy \
    --agent policy \
    --model-path data/1v1 \
    --challenge-user MyName
```

**Random agent:**
```bash
.venv/bin/python -m scripts.play_policy --agent random --challenge-user MyName
```

**Max-power agent:**
```bash
.venv/bin/python -m scripts.play_policy --agent max-power --challenge-user MyName
```

The AI will send you a challenge in the browser. Click **Accept** and play!

---

## Picking specific teams

Use `--ai-team` to control which Pokémon the **AI** uses.  
`--my-team` only **prints** a suggested team for you — you still need to enter it in the Teambuilder yourself.

```bash
.venv/bin/python -m scripts.play_policy \
    --agent random \
    --ai-team steelix \
    --challenge-user MyName
```

**Available named AI teams:** `steelix`, `garchomp`, `conkeldurr`, `rotom_wash`, `corviknight`,
`toxapex`, `excadrill`, `hippowdon`, `breloom`, `volcarona`, `regigigas`, `bellibolt`

To use a different dataset for random sampling:
```bash
.venv/bin/python -m scripts.play_policy --agent random \
    --data-path data/gen9randombattle_db.json \
    --challenge-user MyName
```

---

## All options

| Flag | Default | Description |
|---|---|---|
| `--agent` | `policy` | AI type: `policy`, `random`, or `max-power` |
| `--model-path` | `data/1v1` | Saved `.zip` model path *(policy only)* |
| `--ai-team` | `random` | AI's team — named team or `random` |
| `--my-team` | `random` | Prints a suggested team for you to enter in the Teambuilder |
| `--team-size` | `1` | Number of Pokémon per random team |
| `--data-path` | `data/matchups_gen1ou_db.json` | Dataset used for random team sampling |
| `--format` | `gen1ou` | Battle format |
| `--challenge-user` | **required** | Your Showdown username |
| `--n-challenges` | `1` | How many games to play back-to-back |
| `--player-name` | *(agent type)* | The AI's username on Showdown |
| `--no-verbose` | off | Suppress per-turn BattleState printout |

