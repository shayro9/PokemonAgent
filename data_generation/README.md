# Data Generation

Scripts for generating Pokémon datasets used to train and evaluate the agent.

---

## Files

| File | Purpose |
|---|---|
| `common.js` | Shared helpers and config (format, dedup, type logic, validation) |
| `generate_teams.js` | Generates a pool of individual Pokémon sets (one mon per team) |
| `generate_matchups.js` | Generates winnable 1v1 matchup pairs |

Output files are written to the `data/` folder at the project root.

---

## Prerequisites

### 1. Node.js ≥ 16

Check if you already have it:
```bash
node --version
```

If not, install via [nvm](https://github.com/nvm-sh/nvm) (recommended):
```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
# restart your terminal, then:
nvm install 20
nvm use 20
```

Or download directly from [nodejs.org](https://nodejs.org).

---

### 2. Build `pokemon-showdown`

The scripts import directly from the Showdown simulator's compiled output (`dist/`).
You need to build it once before running any generation script.

```bash
cd pokemon-showdown
npm install
npm run build
cd ..
```

> If you see `Cannot find module '../pokemon-showdown/dist/sim/...'`, the build step was skipped.

---

## Running the Scripts

> **All commands below must be run from the `data_generation/` directory:**
> ```bash
> cd data_generation
> ```

---

### Generate Teams

Generates a pool of individual Pokémon sets from randomly generated Gen 1 OU teams.

```bash
node generate_teams.js
```

Output: `data/teams_gen1ou_db.json`

**Config options** (edit the top of `generate_teams.js`):

| Constant | Default | Description |
|---|---|---|
| `NUM_TO_GENERATE` | `10000` | Number of full teams to attempt generating |
| `MAX_PER_SPECIES` | `60` | Max sets collected per species |
| `ONE_MON_PER_TEAM` | `true` | Pick one random mon per team instead of all 6 |
| `FILTER_SPECIES` | `null` | Only collect sets for a specific species (e.g. `'Starmie'`), or `null` for all |
| `OUTPUT_FILE` | `./data/teams_gen1ou_db.json` | Where to write the output |

---

### Generate Matchups

Generates winnable 1v1 matchup pairs — both sides must have at least one move that can deal damage to the other.

```bash
node generate_matchups.js
```

Output: `data/matchups_gen1ou_db.json`

**Config options** (edit the top of `generate_matchups.js`):

| Constant | Default | Description |
|---|---|---|
| `NUM_TO_GENERATE` | `10000` | Number of valid matchups to collect |
| `FULL_TEAMS` | `false` | `false` = 1v1 (one mon each side), `true` = 6v6 (full teams each side) |
| `OUTPUT_FILE` | `./data/matchups_gen1ou_db.json` | Where to write the output |

---

### Shared Config

Settings that apply to **both** scripts live in `common.js`:

| Constant | Default | Description |
|---|---|---|
| `FORMAT` | `'gen1ou'` | Pokémon Showdown format string |
| `DEDUPE_EXACT_SETS` | `true` | Skip duplicate sets (same species/moves/EVs/etc.) |

---

## Output Format

### `teams_gen1ou_db.json`
```json
{
  "format": "gen1ou",
  "mode": "teams",
  "generatedTeams": 10000,
  "oneMonPerTeam": true,
  "maxPerSpecies": 60,
  "dedupeExactSets": true,
  "poolSize": 8742,
  "pool": [
    { "species": "Starmie", "moves": ["Surf", "Thunderbolt", "Blizzard", "Recover"], ... },
    ...
  ]
}
```

### `matchups_gen1ou_db.json`
```json
{
  "format": "gen1ou",
  "mode": "matchups",
  "matchupCount": 10000,
  "attempts": 11234,
  "filteredUnwinnable": 1234,
  "dedupeExactSets": true,
  "pool": [
    {
      "agent":    [{ "species": "Alakazam", "moves": [...], ... }],
      "opponent": [{ "species": "Snorlax",  "moves": [...], ... }]
    },
    ...
  ]
}
```




