"""
scripts/debug_stat_belief_battle.py
====================================
Real-battle debug script for StatBelief.

Sends a challenge to a human (or bot) opponent, plays with random moves,
and prints the evolving StatBelief after every turn.

Run from the project root:
    python scripts/debug_stat_belief_battle.py

The opponent must accept the challenge in the browser at:
    http://localhost:8000  (or wherever your Showdown server runs)
"""

import asyncio
from types import SimpleNamespace

from poke_env.player import RandomPlayer
from poke_env.battle import Battle, MoveCategory

from combat.stats_belief import (
    StatBelief,
    build_stat_belief,
    level_factor,
    STAT_KEYS,
)
from teams.single_teams import GARCHOMP_TEAM

# ---------------------------------------------------------------------------
# Config — edit these
# ---------------------------------------------------------------------------

CHALLENGE_USER = "shayromelech"   # Showdown username to challenge
BATTLE_FORMAT  = "gen9customgame"
N_CHALLENGES   = 1

# ---------------------------------------------------------------------------
# Per-battle tracker (plain dataclass, no poke-env dependency)
# ---------------------------------------------------------------------------

class BattleBelief:
    """Holds all mutable state for one battle."""

    def __init__(self):
        self.belief: StatBelief | None = None   # initialised on first observe
        self.last_my_hp:  float = 1.0
        self.last_opp_hp: float = 1.0
        self.last_move_category: MoveCategory | None = None
        self.last_move_bp: float = 0.0
        self.last_move_special: bool = False


# ---------------------------------------------------------------------------
# Debug player
# ---------------------------------------------------------------------------

class StatBeliefDebugPlayer(RandomPlayer):
    """Random player that prints StatBelief updates after every turn."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._beliefs: dict[str, BattleBelief] = {}

    # ------------------------------------------------------------------
    # poke-env entry point — called every time we must choose a move
    # ------------------------------------------------------------------

    def choose_move(self, battle: Battle):
        tag = battle.battle_tag
        bb  = self._beliefs.setdefault(tag, BattleBelief())

        me  = battle.active_pokemon
        opp = battle.opponent_active_pokemon

        # ── 1. Initialize belief on the very first turn ─────────────────
        if bb.belief is None:
            bb.belief = build_stat_belief(opp, battle.gen)
            print(f"\n{'='*60}")
            print(f"  Battle started: {tag}")
            print(f"  Our  Pokémon : {me.species}  (lv {me.level})")
            print(f"  Opp  Pokémon : {opp.species} (lv {opp.level})")
            _print_belief("Initial prior", bb.belief)

        # ── 2. Update from the turn that just resolved ───────────────────
        if battle.turn > 0:
            my_hp_now  = me.current_hp_fraction
            opp_hp_now = opp.current_hp_fraction

            damage_dealt    = bb.last_opp_hp - opp_hp_now   # positive = we hurt them
            damage_received = bb.last_my_hp  - my_hp_now    # positive = they hurt us

            lf = level_factor(opp.level)

            # ── Update Def / SpD from damage we dealt ───────────────────
            if damage_dealt > 0.01 and bb.last_move_bp > 0:
                my_attack_stat = (
                    me.stats["spa"] if bb.last_move_special else me.stats["atk"]
                )
                bb.belief = bb.belief.update_from_damage_dealt(
                    damage_fraction=damage_dealt,
                    my_attack=float(my_attack_stat),
                    base_power=bb.last_move_bp,
                    move_is_special=bb.last_move_special,
                    level_factor=lf,
                    modifier=1.0,   # unknown modifier → extra noise absorbed
                    extra_noise_frac=0.10,
                )
                stat_updated = "SpD" if bb.last_move_special else "Def"
                print(f"\n[Turn {battle.turn}] Dealt {damage_dealt*100:.1f}% HP "
                      f"(BP={bb.last_move_bp:.0f}) → updated opp {stat_updated}")

            # ── Update Atk / SpA from damage we received ────────────────
            # We don't know which move the opponent used, so we can only
            # update when we know our relevant defense stat.
            if damage_received > 0.01:
                # Guess special vs physical from opp's revealed moves
                is_special = _guess_opp_move_special(opp)
                def_key    = "spd" if is_special else "def"
                my_def     = me.stats[def_key]
                my_max_hp  = me.max_hp or int(me.current_hp_fraction * 300)

                bb.belief = bb.belief.update_from_damage_received(
                    damage_fraction=damage_received,
                    my_max_hp=float(my_max_hp),
                    my_defense=float(my_def),
                    base_power=75.0,        # unknown BP — use a reasonable default
                    move_is_special=is_special,
                    level_factor=lf,
                    modifier=1.0,
                    extra_noise_frac=0.15,  # higher noise: BP is a guess
                )
                stat_updated = "SpA" if is_special else "Atk"
                print(f"[Turn {battle.turn}] Took {damage_received*100:.1f}% HP "
                      f"(guessed {'special' if is_special else 'physical'}) "
                      f"→ updated opp {stat_updated}")

            # ── Update Spe from speed order ──────────────────────────────
            our_spe = me.stats["spe"]
            we_moved_first = _did_we_move_first(battle)
            if we_moved_first is not None:
                bb.belief = bb.belief.update_from_speed_order(
                    our_spe=float(our_spe),
                    we_moved_first=we_moved_first,
                )
                order = "first" if we_moved_first else "second"
                print(f"[Turn {battle.turn}] We moved {order} "
                      f"(our spe={our_spe}) → updated opp Spe")

            _print_belief(f"Turn {battle.turn} posterior", bb.belief)

        # ── 3. Snapshot HP + chosen move for next turn ───────────────────
        bb.last_my_hp  = me.current_hp_fraction
        bb.last_opp_hp = opp.current_hp_fraction

        order = self.choose_random_move(battle)

        # Record what move we're about to use so next turn can update Def/SpD
        chosen_move = _resolve_chosen_move(battle, order)
        if chosen_move is not None:
            bb.last_move_bp       = float(chosen_move.base_power or 0)
            bb.last_move_special  = (chosen_move.category == MoveCategory.SPECIAL)
            bb.last_move_category = chosen_move.category
        else:
            bb.last_move_bp       = 0.0
            bb.last_move_special  = False
            bb.last_move_category = None

        return order


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_belief(label: str, belief: StatBelief) -> None:
    print(f"\n  ── {label} {'─'*(44 - len(label))}")
    print(f"  {'stat':<5}  {'mean':>7}  {'std':>7}  {'95% CI':>20}")
    for i, key in enumerate(STAT_KEYS):
        mu  = belief.mean[i]
        std = belief.var[i] ** 0.5
        lo, hi = mu - 2 * std, mu + 2 * std
        print(f"  {key:<5}  {mu:>7.1f}  {std:>7.1f}  [{lo:>7.1f}, {hi:>7.1f}]")


def _guess_opp_move_special(opp) -> bool:
    """Heuristic: guess whether the opponent's last move was special.

    Uses the *last revealed move* that is not a status move.
    Falls back to False (physical) when nothing is known.

    :param opp: Opponent Pokémon.
    :returns: ``True`` if guessed special, ``False`` if physical.
    """
    for move in (opp.moves or {}).values():
        if move.category == MoveCategory.SPECIAL:
            return True
        if move.category == MoveCategory.PHYSICAL:
            return False
    return False   # no moves revealed yet


def _did_we_move_first(battle: Battle) -> bool | None:
    """Infer turn order from the opponent's last used move.

    poke-env doesn't expose a direct "who moved first" flag, so we use
    the ``last_request`` heuristic: if the opponent has already used a
    move this turn and their HP has changed, we moved after.

    This is a best-effort approximation — returns ``None`` when ambiguous.

    :param battle: Current battle state.
    :returns: ``True`` if we moved first, ``False`` if second, ``None`` if unknown.
    """
    opp = battle.opponent_active_pokemon
    # If the opponent is in a "must recharge" or "preparing" state, they
    # couldn't have moved, so we effectively moved first.
    if getattr(opp, "preparing", False) or getattr(opp, "must_recharge", False):
        return True
    # Otherwise we don't have reliable turn-order info from the battle object
    # without parsing the raw protocol stream.  Return None to skip the update.
    return None


def _resolve_chosen_move(battle: Battle, order):
    """Extract the Move object corresponding to the order we're about to send.

    :param battle: Current battle state.
    :param order: The order object returned by ``choose_random_move``.
    :returns: The ``Move`` object, or ``None`` if it couldn't be resolved.
    """
    try:
        # poke-env orders have a `move` attribute when they are move orders
        move = getattr(order, "move", None)
        if move is not None:
            return move
    except Exception:
        pass
    # Fallback: return the first available move
    moves = battle.available_moves
    return moves[0] if moves else None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main():
    player = StatBeliefDebugPlayer(
        battle_format=BATTLE_FORMAT,
        team=GARCHOMP_TEAM,
    )

    print(f"Challenging '{CHALLENGE_USER}' to {N_CHALLENGES} battle(s)...")
    await player.send_challenges(CHALLENGE_USER, n_challenges=N_CHALLENGES)
    print("\nAll battles finished.")


if __name__ == "__main__":
    asyncio.run(main())