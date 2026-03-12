"""
scrape_replays.py
=================
Scrape battle replays from https://replay.pokemonshowdown.com and parse
them into structured datasets.

USAGE
-----
Basic scrape:
    python scrape_replays.py --format gen9randombattle --num-replays 500

Custom output directory:
    python scrape_replays.py --format gen9randombattle --num-replays 200 --out-dir data/replays

Resume from a previous run (skip already-fetched IDs):
    python scrape_replays.py --format gen9randombattle --num-replays 1000 --resume

OUTPUT FILES
------------
<out-dir>/
    raw_battles.jsonl          – one JSON object per battle (full log + metadata)

Each battle record has the shape:
    {
        "battle_id":   str,      # e.g. "gen9randombattle-123456789"
        "format":      str,
        "uploadtime":  int,
        "p1":          str,
        "p2":          str,
        "winner":      str,
        "log":         str,      # full battle log
        "inputlog":    str,      # inputlog (needed for supervised learning)
    }

NOTES
-----
* The PS search API returns max 51 results per page; we paginate via the
  `before` (uploadtime) parameter of the last result.
* Rate limiting: we sleep 0.4 s between replay fetches and 0.2 s between
  search pages to be polite to the server.
* Replays whose log cannot be parsed are logged as warnings and skipped.
* For splitting data by action type AFTER building observations, use the
  --split-actions flag in build_supervised_dataset.py instead.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterator

import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEARCH_URL   = "https://replay.pokemonshowdown.com/search.json"
REPLAY_URL   = "https://replay.pokemonshowdown.com/{replay_id}.json"
INPUTLOG_URL = "https://replay.pokemonshowdown.com/{replay_id}.inputlog"

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

def fetch_inputlog(replay_id: str) -> str | None:
    """Download the inputlog for one battle (plain text).

    Only available for formats with autogenerated teams (gen9randombattle,
    gen9doublesrandombattle, …).  Returns ``None`` — without a warning — for
    standard-format replays that never had an inputlog saved.

    :param replay_id: The PS replay ID.
    :returns: Raw inputlog text, or ``None`` if unavailable / on error.
    """
    url = INPUTLOG_URL.format(replay_id=replay_id)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = SESSION.get(url, timeout=20)
            if resp.status_code == 404:
                return None          # no inputlog for this format — silent
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as exc:
            if attempt == MAX_RETRIES:
                print(f"  [WARN] Could not fetch inputlog for {replay_id}: {exc}")
                return None
            time.sleep(2 ** attempt)

    return None


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
    resume: bool,
    fetch_inputlog_flag: bool = True,
) -> None:
    """Run the full scrape pipeline.

    :param format_id: PS battle format, e.g. ``'gen9randombattle'``.
    :param num_replays: Target number of replays to download.
    :param out_dir: Directory to write output files.
    :param resume: If ``True``, skip replays already in ``raw_battles.jsonl``.
    :param fetch_inputlog_flag: If ``True`` (default), also fetch the ``.inputlog``
        for each replay and store it in the ``inputlog`` field.  Required by
        ``build_supervised_dataset.py``.  Adds one extra HTTP request per replay.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "raw_battles.jsonl"

    seen_ids: set[str] = set()
    if resume:
        seen_ids = _load_seen_ids(raw_path)
        print(f"[resume] {len(seen_ids)} replays already on disk — skipping these.")

    total_fetched = 0

    print(f"Scraping format={format_id}  target={num_replays} replays")
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

        # ── Fetch inputlog (needed for supervised dataset) ─────────────
        inputlog_text: str = ""
        if fetch_inputlog_flag:
            time.sleep(FETCH_DELAY)
            inputlog_text = fetch_inputlog(replay_id) or ""

        # ── Store raw battle ───────────────────────────────────────────
        battle_record = {
            "battle_id":  replay_id,
            "format":     format_id,
            "uploadtime": replay.get("uploadtime"),
            "p1":         replay.get("p1"),
            "p2":         replay.get("p2"),
            "winner":     replay.get("winner"),
            "log":        log_text,
            "inputlog":   inputlog_text,   # "" when unavailable or not fetched
        }
        _append_jsonl(raw_path, [battle_record])
        seen_ids.add(replay_id)
        total_fetched += 1

        # ── Progress ──────────────────────────────────────────────────
        if total_fetched % 50 == 0 or total_fetched == num_replays:
            print(f"  [{total_fetched}/{num_replays}] {replay_id}")

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'─'*55}")
    print(f"Done. {total_fetched} replays saved → {raw_path}")
    print(f"  Inputlog fetched: {'yes' if fetch_inputlog_flag else 'no (--no-inputlog)'}")
    print(f"{'─'*55}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scrape Pokémon Showdown replays.",
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
        "--no-inputlog",
        action="store_true",
        default=False,
        help="Skip fetching the .inputlog (halves HTTP requests; dataset builder won't work).",
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
        resume=args.resume,
        fetch_inputlog_flag=not args.no_inputlog,
    )


if __name__ == "__main__":
    main()