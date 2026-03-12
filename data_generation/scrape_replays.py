"""
scrape_replays.py
=================
Scrape battle replays from https://replay.pokemonshowdown.com and parse
them into structured datasets.

USAGE
-----
Basic scrape (all raw battles):
    python scrape_replays.py --format gen9randombattle --num-replays 500

Scrape + split by action type:
    python scrape_replays.py --format gen9randombattle --num-replays 500 --split-actions

Custom output directory:
    python scrape_replays.py --format gen9randombattle --num-replays 200 --out-dir data/replays --split-actions

Resume from a previous run (skip already-fetched IDs):
    python scrape_replays.py --format gen9randombattle --num-replays 1000 --resume

OUTPUT FILES
------------
Without --split-actions:
    <out-dir>/
        raw_battles.jsonl          – one JSON object per battle (full log + metadata)

With --split-actions:
    <out-dir>/
        raw_battles.jsonl          – every battle (as above)
        actions_move.jsonl         – one record per MOVE action taken across all battles
        actions_switch.jsonl       – one record per SWITCH action taken across all battles
        actions_all.jsonl          – every action (move + switch combined)

Each action record has the shape:
    {
        "battle_id":   str,      # e.g. "gen9randombattle-123456789"
        "format":      str,
        "turn":        int,
        "player":      str,      # "p1" or "p2"
        "action_type": str,      # "move" or "switch"
        "action":      str,      # move name or Pokémon being switched in
        "actor":       str,      # slot label, e.g. "p1a: Garchomp"
        "target":      str,      # slot label or "" for self-targeting moves
        "raw_line":    str,      # the verbatim protocol line
    }

NOTES
-----
* The PS search API returns max 51 results per page; we paginate via the
  `before` (uploadtime) parameter of the last result.
* Rate limiting: we sleep 0.4 s between replay fetches and 0.2 s between
  search pages to be polite to the server.
* Replays whose log cannot be parsed are logged as warnings and skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Iterator

import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEARCH_URL = "https://replay.pokemonshowdown.com/search.json"
REPLAY_URL = "https://replay.pokemonshowdown.com/{replay_id}.json"

FETCH_DELAY = 0.4        # seconds between individual replay fetches
SEARCH_DELAY = 0.2       # seconds between search-page requests
MAX_RETRIES = 3
RESULTS_PER_PAGE = 51    # PS returns up to 51; 51st signals another page exists

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "PokemonAgentResearch/1.0 (academic)"})


# ---------------------------------------------------------------------------
# Search / pagination
# ---------------------------------------------------------------------------

def _search_page(format_id: str, before: int | None) -> list[dict]:
    """Fetch one page of search results.

    :param format_id: Battle format string, e.g. ``'gen9randombattle'``.
    :param before: Uploadtime cursor for pagination, or ``None`` for the first page.
    :returns: List of replay metadata dicts from the PS API.
    :raises requests.HTTPError: On non-2xx responses after retries.
    """
    params: dict = {"format": format_id}
    if before is not None:
        params["before"] = before

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = SESSION.get(SEARCH_URL, params=params, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as exc:
            if attempt == MAX_RETRIES:
                raise
            wait = 2 ** attempt
            print(f"  [retry {attempt}/{MAX_RETRIES}] search error: {exc} — waiting {wait}s")
            time.sleep(wait)
    return []


def iter_replay_ids(format_id: str, limit: int) -> Iterator[tuple[str, int]]:
    """Yield ``(replay_id, uploadtime)`` tuples for the given format.

    Paginates through the PS search API until ``limit`` IDs have been
    yielded or there are no more results.

    :param format_id: Battle format string.
    :param limit: Maximum number of IDs to yield.
    :yields: ``(replay_id, uploadtime)`` tuples.
    """
    collected = 0
    before: int | None = None

    while collected < limit:
        results = _search_page(format_id, before)
        if not results:
            break

        # PS uses the 51st entry solely as a "more pages exist" signal
        has_more = len(results) >= RESULTS_PER_PAGE
        page_items = results[:RESULTS_PER_PAGE - 1] if has_more else results

        for item in page_items:
            if collected >= limit:
                return
            replay_id = item.get("id", "")
            uploadtime = item.get("uploadtime", 0)
            if replay_id:
                yield replay_id, uploadtime
                collected += 1

        if not has_more or not page_items:
            break

        before = page_items[-1].get("uploadtime")
        time.sleep(SEARCH_DELAY)


# ---------------------------------------------------------------------------
# Replay fetching
# ---------------------------------------------------------------------------

def fetch_replay(replay_id: str) -> dict | None:
    """Download and return the full replay JSON for one battle.

    :param replay_id: The PS replay ID, e.g. ``'gen9randombattle-123456789'``.
    :returns: The parsed JSON dict, or ``None`` on failure.
    """
    url = REPLAY_URL.format(replay_id=replay_id)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = SESSION.get(url, timeout=20)
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as exc:
            if attempt == MAX_RETRIES:
                print(f"  [WARN] Could not fetch {replay_id}: {exc}")
                return None
            wait = 2 ** attempt
            time.sleep(wait)

    return None


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------

def parse_actions_from_log(log: str, battle_id: str, fmt: str) -> list[dict]:
    """Parse all move/switch actions from a PS battle log string.

    The PS protocol uses lines of the form:
        |move|<actor>|<move_name>|<target>
        |switch|<actor>|<species_detail>|<hp>
        |drag|<actor>|<species_detail>|<hp>       ← treated as switch

    :param log: Full battle log text (the ``log`` field of the replay JSON).
    :param battle_id: Used to populate the ``battle_id`` field of each record.
    :param fmt: Format string, used to populate ``format``.
    :returns: List of action records (dicts) in turn order.
    """
    actions: list[dict] = []
    current_turn = 0

    for raw_line in log.splitlines():
        raw_line = raw_line.strip()
        if not raw_line.startswith("|"):
            continue

        parts = raw_line.lstrip("|").split("|")
        if not parts:
            continue

        tag = parts[0]

        # Track turn number
        if tag == "turn" and len(parts) >= 2:
            try:
                current_turn = int(parts[1])
            except ValueError:
                pass
            continue

        # ── Move ──────────────────────────────────────────────────────
        if tag == "move" and len(parts) >= 3:
            actor = parts[1]            # e.g. "p1a: Garchomp"
            move_name = parts[2]        # e.g. "Earthquake"
            target = parts[3] if len(parts) > 3 else ""
            player = actor[:2] if len(actor) >= 2 else "??"
            actions.append({
                "battle_id":   battle_id,
                "format":      fmt,
                "turn":        current_turn,
                "player":      player,
                "action_type": "move",
                "action":      move_name,
                "actor":       actor,
                "target":      target,
                "raw_line":    raw_line,
            })

        # ── Switch / Drag (forced switch treated the same) ─────────────
        elif tag in ("switch", "drag") and len(parts) >= 3:
            actor = parts[1]            # e.g. "p1a: Garchomp"
            # parts[2] is like "Garchomp, L80, M" — extract just species
            species_detail = parts[2]
            species = species_detail.split(",")[0].strip()
            player = actor[:2] if len(actor) >= 2 else "??"
            actions.append({
                "battle_id":   battle_id,
                "format":      fmt,
                "turn":        current_turn,
                "player":      player,
                "action_type": "switch",
                "action":      species,
                "actor":       actor,
                "target":      "",
                "raw_line":    raw_line,
            })

    return actions


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def _append_jsonl(path: Path, records: list[dict]) -> None:
    """Append ``records`` to a JSONL file (one JSON object per line).

    :param path: File path to append to (created if missing).
    :param records: List of dicts to serialise.
    """
    with path.open("a", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _load_seen_ids(path: Path) -> set[str]:
    """Load already-scraped battle IDs from an existing JSONL file.

    :param path: Path to the ``raw_battles.jsonl`` file (may not exist).
    :returns: Set of ``battle_id`` strings that have been saved before.
    """
    if not path.exists():
        return set()
    seen: set[str] = set()
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                obj = json.loads(line)
                bid = obj.get("battle_id") or obj.get("id")
                if bid:
                    seen.add(bid)
            except json.JSONDecodeError:
                pass
    return seen


# ---------------------------------------------------------------------------
# Main scrape routine
# ---------------------------------------------------------------------------

def scrape(
    format_id: str,
    num_replays: int,
    out_dir: Path,
    split_actions: bool,
    resume: bool,
) -> None:
    """Run the full scrape pipeline.

    :param format_id: PS battle format, e.g. ``'gen9randombattle'``.
    :param num_replays: Target number of replays to download.
    :param out_dir: Directory to write output files.
    :param split_actions: If ``True``, write per-action JSONL files split by type.
    :param resume: If ``True``, skip replays already in ``raw_battles.jsonl``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "raw_battles.jsonl"

    seen_ids: set[str] = set()
    if resume:
        seen_ids = _load_seen_ids(raw_path)
        print(f"[resume] {len(seen_ids)} replays already on disk — skipping these.")

    # Prepare action files
    if split_actions:
        move_path   = out_dir / "actions_move.jsonl"
        switch_path = out_dir / "actions_switch.jsonl"
        all_path    = out_dir / "actions_all.jsonl"
    else:
        move_path = switch_path = all_path = None  # suppress warnings

    total_fetched = 0
    total_actions = {"move": 0, "switch": 0}

    print(f"Scraping format={format_id}  target={num_replays} replays  split_actions={split_actions}")
    print(f"Output directory: {out_dir.resolve()}\n")

    for replay_id, _ in iter_replay_ids(format_id, num_replays + len(seen_ids)):
        if replay_id in seen_ids:
            continue
        if total_fetched >= num_replays:
            break

        time.sleep(FETCH_DELAY)
        replay = fetch_replay(replay_id)
        if replay is None:
            continue

        log_text: str = replay.get("log", "")
        if not log_text:
            print(f"  [WARN] {replay_id}: empty log — skipping")
            continue

        # ── Store raw battle ───────────────────────────────────────────
        battle_record = {
            "battle_id":  replay_id,
            "format":     format_id,
            "uploadtime": replay.get("uploadtime"),
            "p1":         replay.get("p1"),
            "p2":         replay.get("p2"),
            "winner":     replay.get("winner"),
            "log":        log_text,
        }
        _append_jsonl(raw_path, [battle_record])
        seen_ids.add(replay_id)
        total_fetched += 1

        # ── Parse + store actions ─────────────────────────────────────
        if split_actions:
            actions = parse_actions_from_log(log_text, replay_id, format_id)
            move_actions   = [a for a in actions if a["action_type"] == "move"]
            switch_actions = [a for a in actions if a["action_type"] == "switch"]

            if move_actions:
                _append_jsonl(move_path, move_actions)
                total_actions["move"] += len(move_actions)
            if switch_actions:
                _append_jsonl(switch_path, switch_actions)
                total_actions["switch"] += len(switch_actions)
            if actions:
                _append_jsonl(all_path, actions)

        # ── Progress ──────────────────────────────────────────────────
        if total_fetched % 50 == 0 or total_fetched == num_replays:
            msg = f"  [{total_fetched}/{num_replays}] {replay_id}"
            if split_actions:
                msg += f"  (moves={total_actions['move']}, switches={total_actions['switch']})"
            print(msg)

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'─'*55}")
    print(f"Done. {total_fetched} replays saved → {raw_path}")
    if split_actions:
        print(f"  Move    actions : {total_actions['move']:,}  → {move_path}")
        print(f"  Switch  actions : {total_actions['switch']:,}  → {switch_path}")
        print(f"  All     actions : {total_actions['move'] + total_actions['switch']:,}  → {all_path}")
    print(f"{'─'*55}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scrape Pokémon Showdown replays and (optionally) split by action type.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--format",
        default="gen9randombattle",
        help="PS battle format to scrape (default: gen9randombattle).",
    )
    parser.add_argument(
        "--num-replays",
        type=int,
        default=200,
        help="Number of replays to download (default: 200).",
    )
    parser.add_argument(
        "--out-dir",
        default="data/replays",
        help="Output directory for JSONL files (default: data/replays).",
    )
    parser.add_argument(
        "--split-actions",
        action="store_true",
        default=False,
        help="Parse each replay and write separate JSONL files for moves and switches.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Skip replays already present in raw_battles.jsonl.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    scrape(
        format_id=args.format,
        num_replays=args.num_replays,
        out_dir=Path(args.out_dir),
        split_actions=args.split_actions,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
