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


def load_metamon_json(filepath: Path) -> Dict[str, Any]:
    """Load a metamon parsed replay JSON file.

    Args:
        filepath: Path to the JSON file

    Returns:
        Dict containing the parsed battle data
    """
    with lz4.frame.open(filepath, 'r') as f:
        return json.load(f)


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
    # The structure may vary - try multiple possible keys
    observations = (
        battle_data.get("observations") or
        battle_data.get("obs") or
        battle_data.get("states") or
        []
    )

    actions = (
        battle_data.get("actions") or
        battle_data.get("action") or
        []
    )

    metadata = battle_data.get("metadata", {}) or battle_data

    # Get winner if available
    winner = metadata.get("winner", "") or metadata.get("result", "")

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
        if action < 6:  # Assuming 0-5 are switches
            action_type = "switch"
        elif action < 10:  # 6-9 are moves
            action_type = "move"
        else:
            # Unknown action, skip
            continue

        # Convert observation to list if it's a numpy array
        if isinstance(obs, np.ndarray):
            obs_list = obs.tolist()
        elif isinstance(obs, dict):
            # If obs is a dict of arrays, flatten or select main obs
            # Try common keys
            for key in ["observation", "obs", "state", "vector"]:
                if key in obs:
                    obs_value = obs[key]
                    if isinstance(obs_value, np.ndarray):
                        obs_list = obs_value.tolist()
                    else:
                        obs_list = obs_value
                    break
            else:
                # Fallback: use first value
                obs_list = list(obs.values())[0] if obs else []
                if isinstance(obs_list, np.ndarray):
                    obs_list = obs_list.tolist()
        elif isinstance(obs, (list, tuple)):
            obs_list = list(obs)
        else:
            obs_list = [obs]

        samples.append({
            "obs": obs_list,
            "action": action,
            "action_type": action_type,
            "battle_id": battle_id,
            "turn": turn_idx + 1,
            "player": "p1",
            "winner": winner,
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


def main():
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