"""
convert_metamon_json.py
========================
Convert metamon parsed replay JSON files (from HuggingFace) to your supervised
learning format.

This version automatically searches for JSON files in subdirectories!

USAGE
-----
python convert_metamon_json.py \
    --input-dir data/metamon/gen1ou \
    --output data/supervised/gen1ou.npz

OUTPUT FORMAT
-------------
Output is always a compressed NumPy archive (.npz).  Arrays stored:
  obs          : float32 (N, obs_dim) — observation vectors
  actions      : int16   (N,)         — action indices 0-9
  episode_idx  : int32   (N,)         — maps each step to an episode
  turns        : int16   (N,)         — turn number within episode
  outcomes     : float32 (N,)         — NaN except terminal step (+1/-1)
"""

import argparse
import json
import sys
import lz4.frame

from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

# Allow `python data_generation/convert_metamon_json.py` from the project root
# in addition to `python -m data_generation.convert_metamon_json`.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from data_generation.metamon_state_gen1 import metamon_to_obs_gen1
from env.battle_config import BattleConfig


def load_metamon_json(filepath: Path) -> Dict[str, Any]:
    """Load a metamon parsed replay JSON file.

    Args:
        filepath: Path to the JSON file

    Returns:
        Dict containing the parsed battle data
    """
    with lz4.frame.open(filepath, 'r') as f:
        return json.load(f)


def _convert_metamon_state_6v6(state: Dict[str, Any], turn: int) -> np.ndarray:
    """Convert a single metamon state dict to the 6v6 Gen-1 observation vector.

    Uses the existing ``BattleStateGen1`` pipeline via proxy adapters, producing
    a vector whose length matches ``BattleConfig.gen1().obs_dim``.

    Args:
        state: One element from the 'states' array in metamon JSON.
        turn:  Zero-based turn index.

    Returns:
        np.ndarray matching the current BattleStateGen1.to_array() format.
    """
    return metamon_to_obs_gen1(state, turn)


def convert_metamon_battle(
    battle_data: Dict[str, Any],
    battle_id: str,
) -> List[Dict[str, Any]]:
    """Convert a single metamon battle to list of state in the project format.

    Args:
        battle_data:  Metamon parsed battle data.
        battle_id:    Battle identifier.

    Returns:
        List of state in the project format.
    """
    samples = []

    # Extract sequences from metamon format
    observations = battle_data.get("states")
    actions = battle_data.get("actions")

    # Skip if no valid data
    if not observations or not actions:
        return []

    _obs_dim = BattleConfig.gen1().obs_dim
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

        if 4 <= action <= 9:  # 4-9 are switches
            action = action - 4
        elif 0 <= action <= 3:  # 0-3 are moves
            action = action + 6
        else:
            continue

        state_arr = _convert_metamon_state_6v6(obs, turn_idx)
        if state_arr.shape[0] != _obs_dim:
            raise ValueError(
                f"Obs shape mismatch at turn {turn_idx}: "
                f"expected {_obs_dim} dims, got {state_arr.shape[0]}"
            )

        samples.append({
            "obs": state_arr,       # ndarray — callers convert to list for jsonl
            "action": action,
            "battle_id": battle_id,
            "turn": turn_idx,
        })

    if samples:
        outcome = 1 if observations[-1].get('battle_won', False) else -1
        samples[-1]["outcome"] = outcome

    return samples


def convert_dataset(
    input_dir: Path,
    output_path: Path,
    max_battles: Optional[int] = None,
) -> None:
    """Convert metamon JSON files to a compressed NumPy archive (.npz).

    Args:
        input_dir:    Directory containing metamon JSON files (or parent).
        output_path:  Output path — must have a .npz extension.
        max_battles:  Maximum number of battles to process (None = all).
    """
    if output_path.suffix != ".npz":
        print(f"ERROR: Output path must have a .npz extension, got: {output_path}")
        sys.exit(1)

    json_files = list(input_dir.rglob("*.json.lz4"))

    if not json_files:
        print(f"ERROR: No JSON files found in {input_dir}")
        print("Did you extract the tar.gz file?")
        sys.exit(1)

    print(f"Found {len(json_files)} battle files in {input_dir}")

    if max_battles:
        json_files = json_files[:max_battles]
        print(f"Processing first {len(json_files)} battles")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_samples = 0
    failed_count  = 0
    empty_count   = 0

    all_obs:     List[np.ndarray] = []
    all_actions: List[int]        = []
    all_ep_idx:  List[int]        = []
    all_turns:   List[int]        = []
    all_outcomes: List[float]     = []
    episode_counter = 0

    for idx, json_file in enumerate(json_files):
        if (idx + 1) % 100 == 0:
            print(f"  [{idx + 1}/{len(json_files)}] battles processed | "
                  f"samples: {total_samples}")
        try:
            battle_data = load_metamon_json(json_file)
            battle_id   = Path(json_file.stem).stem

            samples = convert_metamon_battle(battle_data, battle_id)

            if not samples:
                empty_count += 1
                continue

            for s in samples:
                all_obs.append(s["obs"])
                all_actions.append(s["action"])
                all_ep_idx.append(episode_counter)
                all_turns.append(s["turn"])
                all_outcomes.append(float(s.get("outcome", float("nan"))))
                total_samples += 1
            episode_counter += 1

        except Exception as e:
            failed_count += 1
            if failed_count <= 3:
                print(f"  ⚠ Failed to process {json_file.name}: {e}")

    if all_obs:
        print("  Compressing and writing .npz …")
        np.savez_compressed(
            output_path,
            obs         = np.stack(all_obs).astype(np.float32),
            actions     = np.array(all_actions, dtype=np.int16),
            episode_idx = np.array(all_ep_idx,  dtype=np.int32),
            turns       = np.array(all_turns,   dtype=np.int16),
            outcomes    = np.array(all_outcomes, dtype=np.float32),
        )

    _print_summary(
        len(json_files), failed_count, empty_count,
        total_samples, output_path,
    )


def _print_summary(
    n_files: int,
    failed: int,
    empty: int,
    total: int,
    output_path: Path,
) -> None:
    print(f"\n{'─'*60}")
    print(f"Battles processed : {n_files - failed - empty}")
    print(f"Empty battles     : {empty}")
    print(f"Failed            : {failed}")
    print(f"Total samples     : {total}")
    if output_path.exists():
        size_mb = output_path.stat().st_size / 1_048_576
        print(f"Output            : {output_path}  ({size_mb:.1f} MB)")
    else:
        print(f"Output            : {output_path}")
    print(f"{'─'*60}")

    if failed > 0:
        print(f"\n⚠ {failed} battles failed to process.")
        print("This may indicate a mismatch between the expected and actual")
        print("metamon JSON structure. Check the first few error messages above.")
    if empty > 0:
        print(f"\n⚠ {empty} battles were empty (no valid observations/actions).")

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
        help="Output path — must have a .npz extension",
    )
    parser.add_argument(
        "--max-battles",
        type=int,
        help="Maximum number of battles to process (default: all)",
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
        max_battles=args.max_battles,
    )


if __name__ == "__main__":
    main()