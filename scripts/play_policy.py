"""
play_policy.py  --  Challenge a human on the local Showdown server with any agent.

Usage
-----
From the repo root:

    # Trained policy, both teams randomly sampled from the gen1ou dataset (defaults)
    python -m scripts.play_policy --agent policy --model-path data/1v1 --challenge-user MyName

    # Trained policy with fixed teams
    python -m scripts.play_policy --agent policy --model-path data/1v1 \
        --ai-team steelix --my-team garchomp --challenge-user MyName

    # Random agent, both teams from the dataset
    python -m scripts.play_policy --agent random --challenge-user MyName

    # Max-power agent, fixed AI team, random human team
    python -m scripts.play_policy --agent max-power --ai-team volcarona --challenge-user MyName

Open http://localhost:8000 in your browser and log in as --challenge-user to accept.
"""

import argparse
import asyncio

from poke_env import (
    LocalhostServerConfiguration,
    AccountConfiguration,
    RandomPlayer,
    MaxBasePowerPlayer,
)

from env.singles_env_wrapper import PokemonRLWrapper
from agents.policy_player import PolicyPlayer
from teams.generators import matchup_generator
from data.prossesing import load_pool

AGENT_CHOICES = ["policy", "random", "max-power"]
DEFAULT_FORMAT = "gen1ou"
DEFAULT_DATA_PATH = "data/matchups_gen1ou_db.json"

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

def _resolve_team(team_arg, pool, side, team_size):
    """Return a packed team string.
    'random' -> sample team_size Pokemon from pool using the given side key.
    Named string -> look up in TEAM_BY_NAME (single Pokemon, team_size ignored).
    """
    team = next(matchup_generator(pool=pool))[0 if side == "agent" else 1]
    return team


def _print_packed_team(label, packed):
    """Pretty-print a single packed Showdown team slot."""
    SEP = "=" * 55
    print(f"\n{SEP}")
    print(f"  {label}")
    print(SEP)

    slots = [s.strip() for s in packed.strip().split("]]") if s.strip()]
    for slot in slots:
        # Export/text format (newline-separated) -- print as-is
        if "\n" in slot:
            print(slot.strip())
            print()
            continue

        parts = slot.split("|")
        nickname  = parts[0]  if len(parts) > 0  else "?"
        species   = parts[1]  if len(parts) > 1  else ""
        item      = parts[2]  if len(parts) > 2  else ""
        ability   = parts[3]  if len(parts) > 3  else ""
        moves_raw = parts[4]  if len(parts) > 4  else ""
        nature    = parts[5]  if len(parts) > 5  else ""
        evs_raw   = parts[6]  if len(parts) > 6  else ""
        ivs_raw   = parts[8]  if len(parts) > 8  else ""
        level     = parts[10] if len(parts) > 10 else "100"

        display_name = nickname if not species else f"{nickname} ({species})"
        stat_keys = ["HP", "Atk", "Def", "SpA", "SpD", "Spe"]

        def _fmt(raw):
            vals = raw.split(",") if raw else []
            pairs = [f"{k}: {v}" for k, v in zip(stat_keys, vals) if v and v != "0"]
            return ", ".join(pairs) if pairs else "-"

        moves = [m.strip() for m in moves_raw.split(",") if m.strip()]

        print(f"  Pokemon  : {display_name}")
        if item:    print(f"  Item     : {item}")
        if ability: print(f"  Ability  : {ability}")
        if nature:  print(f"  Nature   : {nature}")
        print(      f"  Level    : {level or 100}")
        if evs_raw: print(f"  EVs      : {_fmt(evs_raw)}")
        if ivs_raw: print(f"  IVs      : {_fmt(ivs_raw)}")
        if moves:   print(f"  Moves    : {', '.join(moves)}")
        print()

    print(SEP)


# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------

def build_arg_parser():
    team_choices = ["random"]

    p = argparse.ArgumentParser(description="Challenge a human with a chosen AI agent.")

    p.add_argument("--agent", default="policy", choices=AGENT_CHOICES,
                   help="AI type: 'policy', 'random', or 'max-power'. Default: policy.")
    p.add_argument("--model-path", default="data/1v1",
                   help="Path to the saved .zip model (policy only). Default: data/1v1.")
    p.add_argument("--ai-team", default="random", choices=team_choices,
                   metavar="{random, <team-name>}",
                   help="AI's team: a named team or 'random' (sampled from dataset). Default: random.")
    p.add_argument("--my-team", default="random", choices=team_choices,
                   metavar="{random, <team-name>}",
                   help="Your team: a named team or 'random'. Default: random.")
    p.add_argument("--data-path", default=DEFAULT_DATA_PATH,
                   help=f"Dataset used for random team sampling. Default: {DEFAULT_DATA_PATH}.")
    p.add_argument("--format", default=DEFAULT_FORMAT,
                   help=f"Battle format. Default: {DEFAULT_FORMAT}.")
    p.add_argument("--challenge-user", required=True,
                   help="Showdown username to challenge (you).")
    p.add_argument("--n-challenges", type=int, default=1,
                   help="Number of challenges to send. Default: 1.")
    p.add_argument("--player-name", default=None,
                   help="AI's Showdown username. Default: '<AgentType>Bot'.")
    p.add_argument("--team-size", type=int, default=1,
                   help="Number of Pokemon per team when using 'random'. Default: 1.")
    p.add_argument("--no-verbose", action="store_true", default=False,
                   help="Suppress per-turn BattleState output (verbose is ON by default).")
    return p


# ------------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------------

async def main():
    args = build_arg_parser().parse_args()
    verbose = not args.no_verbose

    needs_pool = args.ai_team == "random" or args.my_team == "random"
    pool = load_pool(args.data_path) if needs_pool else []

    ai_team = _resolve_team(args.ai_team, pool, side="agent",    team_size=args.team_size)
    my_team = _resolve_team(args.my_team, pool, side="opponent", team_size=args.team_size)

    player_name = args.player_name or args.agent.capitalize() + "Bot"

    # Print both teams before connecting
    _print_packed_team(f"AI Team  ({player_name})", ai_team)
    _print_packed_team(f"Suggested Team for YOU ({args.challenge_user})  <-- enter this in Teambuilder", my_team)

    shared = dict(
        battle_format=args.format,
        team=ai_team,
        server_configuration=LocalhostServerConfiguration,
        account_configuration=AccountConfiguration(player_name, None),
    )

    if args.agent == "policy":
        wrapper = PokemonRLWrapper(
            battle_format=args.format,
            team=ai_team,
            server_configuration=LocalhostServerConfiguration,
            account_configuration1=AccountConfiguration(f"{player_name}_env1", None),
            account_configuration2=AccountConfiguration(f"{player_name}_env2", None),
            strict=False,
        )
        player = PolicyPlayer(
            model_path=args.model_path,
            wrapper=wrapper,
            verbose=verbose,
            **shared,
        )
    elif args.agent == "random":
        player = RandomPlayer(**shared)
    elif args.agent == "max-power":
        player = MaxBasePowerPlayer(**shared)

    print(f"\nSending {args.n_challenges} challenge(s) to '{args.challenge_user}'...")
    print("Open http://localhost:8000, log in, and accept the challenge.")
    await player.send_challenges(args.challenge_user, n_challenges=args.n_challenges)
    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
