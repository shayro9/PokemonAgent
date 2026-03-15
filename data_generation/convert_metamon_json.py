"""
convert_metamon_json_v2.py
===========================
Convert metamon parsed replay JSON files (from HuggingFace) to your supervised
learning format.

This version automatically searches for JSON files in subdirectories!

USAGE
-----
# Auto-find JSON files in data/metamon/gen9ou or subdirectories
python convert_metamon_json_v2.py \
    --input-dir data/metamon/gen9ou \
    --output data/supervised/gen9ou.jsonl \
    --split-actions

# Specify exact directory if you know it
python convert_metamon_json_v2.py \
    --input-dir data/metamon/gen9ou/gen9ou \
    --output data/supervised/gen9ou.jsonl \
    --no-auto-search

OUTPUT FORMAT
-------------
Each line in the output JSONL file is a single timestep:
{
    "obs": [float, ...],      # Observation vector
    "action": int,            # Action index (0-9 in your encoding)
    "action_type": "move",    # "move" or "switch"
    "battle_id": str,         # Battle identifier
    "turn": int,              # Turn number
    "player": "p1",           # Player perspective
    "winner": str,            # Winner username (if available)
}
"""

import argparse
import json
import sys
import lz4.frame

from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from six import moves

from env.embed import (
    embed_status, embed_weather, embed_effects, calc_types_vector, embed_move,
    Weather, Status, Effect, PokemonType, Move,
    MOVE_EMBED_LEN, MAX_MOVES,
)

from combat.stats_belief import build_raw_stat_belief

GEN = 1


def load_metamon_json(filepath: Path) -> Dict[str, Any]:
    """Load a metamon parsed replay JSON file.

    Args:
        filepath: Path to the JSON file

    Returns:
        Dict containing the parsed battle data
    """
    with lz4.frame.open(filepath, 'r') as f:
        return json.load(f)


def convert_metamon_state_to_vector(state: Dict[str, Any], turn) -> np.ndarray:
    """Convert a single metamon state dict to your 383-dim observation vector.

    Args:
        :param state: One element from the 'states' array in metamon JSON
        :param turn: the index of the state

    Returns:
        np.ndarray of shape (383,) matching your BattleState.to_array() format
    """
    my_poke = state["player_active_pokemon"]
    opp_poke = state["opponent_active_pokemon"]

    vec = []

    # Turn (1 dim) - not in metamon, use 0
    vec.append(turn)

    # Weather (9 dims)
    weather_vec = embed_weather(getattr(Weather, state["weather"], None))
    vec.extend(weather_vec)
    # My HP (1 dim)
    vec.append(my_poke.get("hp_pct", 1.0))

    # My stats (6 dims: HP, Atk, Def, SpA, SpD, Spe)
    my_stats = np.array([
        my_poke.get("base_hp", 100),
        my_poke.get("base_atk", 100),
        my_poke.get("base_def", 100),
        my_poke.get("base_spa", 100),
        my_poke.get("base_spd", 100),
        my_poke.get("base_spe", 100),
    ], dtype=np.float32) / 512.0
    vec.extend(np.minimum(my_stats, 1.0))

    # My boosts (7 dims)
    my_boosts = np.array([
        my_poke.get("atk_boost", 0),
        my_poke.get("def_boost", 0),
        my_poke.get("spa_boost", 0),
        my_poke.get("spd_boost", 0),
        my_poke.get("spe_boost", 0),
        my_poke.get("accuracy_boost", 0),
        my_poke.get("evasion_boost", 0),
    ], dtype=np.float32) / 6.0
    vec.extend(my_boosts)

    # My status (7 dims)
    my_status = getattr(Status, my_poke.get("status", "nostatus"), None)
    status_vec = embed_status(my_status)
    vec.extend(status_vec)

    # My effects (3 dims)
    my_effects = getattr(Effect, my_poke.get("effect", "noeffect"), [])
    effects_vec = embed_effects(my_effects)
    vec.extend(effects_vec)

    # My STAB (1 dim) - not in metamon, use 1.0
    vec.append(my_poke.get("stab_multiplier", 1.5) / 2.0)

    # Opp HP (1 dim)
    vec.append(opp_poke.get("hp_pct", 1.0))

    # Opp stat belief (12 dims: 6 means + 6 stds)
    # Metamon doesn't have belief, use base stats as mean with 0 std
    opp_stats_belief = build_raw_stat_belief(opp_poke, opp_poke["lvl"], GEN).to_array()
    vec.extend(opp_stats_belief)

    # Opp boosts (7 dims)
    opp_boosts = np.array([
        opp_poke.get("atk_boost", 0),
        opp_poke.get("def_boost", 0),
        opp_poke.get("spa_boost", 0),
        opp_poke.get("spd_boost", 0),
        opp_poke.get("spe_boost", 0),
        opp_poke.get("accuracy_boost", 0),
        opp_poke.get("evasion_boost", 0),
    ], dtype=np.float32) / 6.0
    vec.extend(opp_boosts)

    # Opp status (7 dims)
    opp_status = getattr(Status, opp_poke.get("status", "nostatus"), None)
    opp_status_vec = embed_status(opp_status)
    vec.extend(opp_status_vec)

    # Opp effects (3 dims)
    opp_effects = getattr(Effect, opp_poke.get("effect", "noeffect"), [])
    opp_effects_vec = embed_effects(opp_effects)
    vec.extend(opp_effects_vec)

    # Opp preparing, STAB, is_terastallized (3 dims)
    vec.append(0.0)  # preparing - not in metamon
    vec.append(1.5 / 2.0)  # opp_stab
    vec.append(0.0)  # is_terastallized

    # Opp tera multiplier (2 dims)
    opp_tera_type = getattr(PokemonType, opp_poke["tera_type"], None)
    my_raw_types = [getattr(PokemonType, t.upper(), None) for t in my_poke["types"].split()]
    opp_raw_types = [getattr(PokemonType, t.upper(), None) for t in opp_poke["types"].split()]

    my_types = list(filter(lambda x: x is not None, my_raw_types))
    opp_types = list(filter(lambda x: x is not None, opp_raw_types))
    vec.extend(calc_types_vector(my_types, opp_tera_type, GEN, True))


    # My moves (MAX_MOVES * MOVE_EMBED_LEN dims)
    my_moves_vec = np.zeros(MAX_MOVES * MOVE_EMBED_LEN, dtype=np.float32)
    for i, move_dict in enumerate(my_poke.get("moves", [])[:MAX_MOVES]):
        move = Move(move_id=move_dict["name"], gen=GEN)
        move_emb = embed_move(move, opp_types, my_types, GEN)
        my_moves_vec[i * MOVE_EMBED_LEN:(i + 1) * MOVE_EMBED_LEN] = move_emb
    vec.extend(my_moves_vec)

    # Opp moves (MAX_MOVES * MOVE_EMBED_LEN dims)
    opp_moves_vec = np.zeros(MAX_MOVES * MOVE_EMBED_LEN, dtype=np.float32)
    for i, move_dict in enumerate(opp_poke.get("moves", [])[:MAX_MOVES]):
        move = Move(move_id=move_dict["name"], gen=GEN)
        move_emb = embed_move(move, my_types, opp_types, GEN)
        opp_moves_vec[i * MOVE_EMBED_LEN:(i + 1) * MOVE_EMBED_LEN] = move_emb
    vec.extend(opp_moves_vec)

    # Opp protect belief (1 dim)
    vec.append(0.0)

    result = np.array(vec, dtype=np.float32)
    assert len(result) == 383, f"Expected 383 dims, got {len(result)}"
    return result


def convert_metamon_battle(
    battle_data: Dict[str, Any],
    battle_id: str,
) -> List[Dict[str, Any]]:
    """Convert a single metamon battle to your sample format.

    Args:
        battle_data: Metamon parsed battle data
        battle_id: Battle identifier

    Returns:
        List of sample dicts in your format
    """
    samples = []

    # Extract sequences from metamon format
    observations = battle_data.get("states")
    actions = battle_data.get("actions")

    # Skip if no valid data
    if not observations or not actions:
        return []

    # Process each timestep
    for turn_idx, (obs, action) in enumerate(zip(observations, actions)):
        # Skip if action is missing/invalid
        if action is None or (isinstance(action, (int, float)) and action < 0):
            continue

        # Convert action to int if needed
        if isinstance(action, dict):
            action = action.get("action", action.get("index", -1))

        action = int(action)

        if action < 0:
            continue

        # Determine action type based on action index
        # This is a heuristic - adjust based on your action space
        if action > 3:  # Assuming 4-8 are switches
            action_type = "switch"
        elif 0 <= action <= 3:  # 0-3 are moves
            action_type = "move"
        else:
            # skip
            continue

        state = convert_metamon_state_to_vector(obs, turn_idx)

        state_list = state.tolist()
        samples.append({
            "obs": state_list,
            "action": action,
            "action_type": action_type,
            "battle_id": battle_id,
            "player": "p1",
        })

    return samples


def convert_dataset(
    input_dir: Path,
    output_path: Path,
    split_actions: bool = False,
    max_battles: Optional[int] = None,
    auto_search: bool = True,
) -> None:
    """Convert metamon JSON files to your format.

    Args:
        input_dir: Directory containing metamon JSON files (or parent)
        output_path: Output JSONL path
        split_actions: If True, create separate files for moves/switches
        max_battles: Maximum number of battles to process (None = all)
        auto_search: If True, automatically search subdirectories for JSON files
    """
    # Find all JSON files
    json_files = list(input_dir.rglob("*.json.lz4"))

    if not json_files:
        print(f"ERROR: No JSON files found in {input_dir}")
        print("Did you extract the tar.gz file?")
        sys.exit(1)

    print(f"Found {len(json_files)} battle files in {input_dir}")

    # Limit number of files if requested
    if max_battles:
        json_files = json_files[:max_battles]
        print(f"Processing first {len(json_files)} battles")

    # Prepare output files
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_fh = output_path.open("w", encoding="utf-8")

    move_fh = None
    switch_fh = None
    if split_actions:
        move_path = output_path.parent / f"{output_path.stem}_move{output_path.suffix}"
        switch_path = output_path.parent / f"{output_path.stem}_switch{output_path.suffix}"
        move_fh = move_path.open("w", encoding="utf-8")
        switch_fh = switch_path.open("w", encoding="utf-8")

    # Convert each battle
    total_samples = 0
    action_counts = {"move": 0, "switch": 0}
    failed_count = 0
    empty_count = 0

    try:
        for idx, json_file in enumerate(json_files):
            if (idx + 1) % 100 == 0:
                print(f"  [{idx + 1}/{len(json_files)}] battles processed | "
                      f"samples: {total_samples} "
                      f"(moves={action_counts['move']}, switches={action_counts['switch']})")

            try:
                # Load battle data
                battle_data = load_metamon_json(json_file)
                battle_id = json_file.stem  # Use filename as battle ID

                # Convert to samples
                samples = convert_metamon_battle(battle_data, battle_id)

                if not samples:
                    empty_count += 1
                    continue

                # Write samples
                for sample in samples:
                    action_counts[sample["action_type"]] += 1
                    total_samples += 1

                    sample_json = json.dumps(sample) + "\n"
                    out_fh.write(sample_json)

                    if split_actions:
                        if sample["action_type"] == "move":
                            move_fh.write(sample_json)
                        else:
                            switch_fh.write(sample_json)

            except Exception as e:
                failed_count += 1
                if failed_count <= 3:  # Only show first 3 errors
                    print(f"  ⚠ Failed to process {json_file.name}: {e}")

        print(f"\n{'─'*60}")
        print(f"Battles processed : {len(json_files) - failed_count - empty_count}")
        print(f"Empty battles     : {empty_count}")
        print(f"Failed            : {failed_count}")
        print(f"Total samples     : {total_samples}")
        print(f"  Move samples    : {action_counts['move']}")
        print(f"  Switch samples  : {action_counts['switch']}")
        print(f"Output            : {output_path}")
        if split_actions:
            print(f"  Move file       : {move_path}")
            print(f"  Switch file     : {switch_path}")
        print(f"{'─'*60}")

        if failed_count > 0:
            print(f"\n⚠ {failed_count} battles failed to process.")
            print("This may indicate a mismatch between the expected and actual")
            print("metamon JSON structure. Check the first few error messages above.")

        if empty_count > 0:
            print(f"\n⚠ {empty_count} battles were empty (no valid observations/actions).")

    finally:
        out_fh.close()
        if move_fh:
            move_fh.close()
        if switch_fh:
            switch_fh.close()

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert metamon JSON files to supervised learning format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing metamon JSON files (will auto-search subdirectories)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSONL path (e.g., data/supervised/gen9ou.jsonl)",
    )
    parser.add_argument(
        "--split-actions",
        action="store_true",
        help="Create separate files for move and switch actions",
    )
    parser.add_argument(
        "--max-battles",
        type=int,
        help="Maximum number of battles to process (default: all)",
    )
    parser.add_argument(
        "--no-auto-search",
        action="store_true",
        help="Don't automatically search subdirectories for JSON files",
    )

    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if not args.input_dir.exists():
        print(f"ERROR: Input directory does not exist: {args.input_dir}")
        print("\nDid you run download_metamon_hf.py first?")
        sys.exit(1)

    convert_dataset(
        input_dir=args.input_dir,
        output_path=args.output,
        split_actions=args.split_actions,
        max_battles=args.max_battles,
        auto_search=not args.no_auto_search,
    )


if __name__ == "__main__":
    main()