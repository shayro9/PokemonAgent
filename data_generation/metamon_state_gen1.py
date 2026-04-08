from __future__ import annotations

import warnings
from functools import lru_cache
from typing import Optional

import numpy as np

from poke_env.battle import Move
from poke_env.battle.effect import Effect
from poke_env.battle.pokemon_type import PokemonType
from poke_env.battle.side_condition import SideCondition
from poke_env.battle.status import Status

from env.states.gen1.battle_state_gen_1 import BattleStateGen1

# ---------------------------------------------------------------------------
# Mapping tables
# ---------------------------------------------------------------------------

_STATUS_MAP: dict[str, Status] = {
    'brn': Status.BRN,
    'frz': Status.FRZ,
    'par': Status.PAR,
    'psn': Status.PSN,
    'slp': Status.SLP,
    'tox': Status.TOX,
    'fnt': Status.FNT,
}

_EFFECT_MAP: dict[str, Effect] = {
    'confusion': Effect.CONFUSION,
    'encore':    Effect.ENCORE,
}

_SIDE_CONDITION_MAP: dict[str, SideCondition] = {
    'reflect':      SideCondition.REFLECT,
    'lightscreen':  SideCondition.LIGHT_SCREEN,
    'light_screen': SideCondition.LIGHT_SCREEN,
}


# ---------------------------------------------------------------------------
# Move cache
# ---------------------------------------------------------------------------

@lru_cache(maxsize=4096)
def _cached_move(move_id: str, gen: int) -> Move:
    """Instantiate and cache a poke-env Move object keyed by (id, gen)."""
    return Move(move_id, gen=gen)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_types(types_str: str) -> list:
    """Convert a space-separated type string to a list of PokemonType enums.

    :param types_str: e.g. ``'normal'``, ``'water ice'``, ``'notype'``.
    :returns: List of ``PokemonType`` enums; ``[None]`` if none resolved.
    """
    out = []
    for token in types_str.split():
        if token.lower() == 'notype':
            continue
        try:
            out.append(PokemonType[token.upper()])
        except KeyError:
            pass
    return out if out else [None]


def _parse_side_conditions(conditions: list) -> dict:
    """Map a list of condition name strings to a ``{SideCondition: 1}`` dict.

    :param conditions: List of condition name strings from the metamon state.
    :returns: Dict keyed by ``SideCondition`` enums.
    """
    result: dict[SideCondition, int] = {}
    for cond in conditions:
        if not isinstance(cond, str):
            continue
        key = cond.lower().replace(' ', '_')
        sc = _SIDE_CONDITION_MAP.get(key)
        if sc is not None:
            result[sc] = 1
    return result


# ---------------------------------------------------------------------------
# Proxy classes
# ---------------------------------------------------------------------------

class MetamonPokemonProxy:
    """
    Wraps a metamon pokemon dict to expose the attribute interface expected by
    ``MyPokemonStateGen1`` and ``OpponentPokemonStateGen1``.

    Parameters
    ----------
    poke_dict  : Raw pokemon dict from the metamon state.
    is_active  : Whether this pokemon is the active battler.
    is_fainted : Force-mark as fainted (overrides hp_pct check).
    gen        : Pokémon generation (default 1).
    """

    def __init__(
        self,
        poke_dict: dict,
        is_active: bool = False,
        is_fainted: bool = False,
        gen: int = 1,
    ) -> None:
        hp_pct  = float(poke_dict.get('hp_pct', 1.0))
        species = poke_dict.get('name') or poke_dict.get('base_species') or 'none'

        self.current_hp_fraction: float = hp_pct
        self.species: str = species

        # Gen 1 uses "spc" for the combined special stat; metamon stores "base_spa"
        self.stats: dict[str, float] = {
            'hp':  float(poke_dict.get('base_hp',  100)),
            'atk': float(poke_dict.get('base_atk', 100)),
            'def': float(poke_dict.get('base_def', 100)),
            'spc': float(poke_dict.get('base_spa', 100)),
            'spe': float(poke_dict.get('base_spe', 100)),
        }
        # OpponentPokemonStateGen1.estimate_stats reads pokemon.base_stats
        self.base_stats: dict[str, float] = self.stats

        self.boosts: dict[str, float] = {
            'atk':      float(poke_dict.get('atk_boost',      0)),
            'def':      float(poke_dict.get('def_boost',      0)),
            'spa':      float(poke_dict.get('spa_boost',      0)),
            'spe':      float(poke_dict.get('spe_boost',      0)),
            'accuracy': float(poke_dict.get('accuracy_boost', 0)),
            'evasion':  float(poke_dict.get('evasion_boost',  0)),
        }

        types_str = str(poke_dict.get('types', 'normal') or 'normal')
        self.types: list = _parse_types(types_str)

        status_str = str(poke_dict.get('status', 'nostatus') or 'nostatus').lower()
        self.status: Optional[Status] = _STATUS_MAP.get(status_str, None)

        effect_str = str(poke_dict.get('effect', 'noeffect') or 'noeffect').lower()
        self.effects: set[Effect] = (
            {_EFFECT_MAP[effect_str]} if effect_str in _EFFECT_MAP else set()
        )

        # STAB cannot be determined from replay data alone (we don't have the
        # attacking move at state-construction time); the state encoder treats 0.0
        # as "unknown" and handles it gracefully.
        self.stab_multiplier: float = 0.0
        self.active:  bool = is_active
        self.fainted: bool = (hp_pct == 0.0 and species != 'none') or is_fainted

        # Build {move_id: Move} dict using the shared move cache
        self.moves: dict[str, Move] = {}
        for move_d in poke_dict.get('moves', []):
            move_id = move_d.get('name', '')
            if not move_id:
                continue
            try:
                self.moves[move_id] = _cached_move(move_id, gen)
            except Exception as e:
                warnings.warn(f"Skipping unknown move '{move_id}' for '{species}': {e}")

        # Attributes used exclusively by OpponentPokemonStateGen1
        self.preparing:       float = 0.0
        self.must_recharge:   float = 0.0
        self.protect_counter: float = -1.0


class MetamonBattleProxy:
    """
    Wraps a metamon state dict to expose the interface expected by
    ``BattleStateGen1``.

    Pass directly to ``BattleStateGen1``::

        obs = BattleStateGen1(MetamonBattleProxy(state, turn)).to_array()

    Parameters
    ----------
    state : Single state dict from a metamon battle file (``d['states'][i]``).
    turn  : Zero-based turn index for this state.
    gen   : Pokémon generation (default 1).
    """

    def __init__(self, state: dict, turn: int, gen: int = 1) -> None:
        missing = [k for k in ('player_active_pokemon', 'opponent_active_pokemon') if k not in state]
        if missing:
            raise KeyError(
                f"Metamon state dict is missing required keys: {missing}. "
                f"Present keys: {list(state.keys())}"
            )

        my_active_dict  = state['player_active_pokemon']
        opp_active_dict = state['opponent_active_pokemon']
        bench_dicts     = state.get('available_switches', [])

        self.active_pokemon           = MetamonPokemonProxy(my_active_dict,  is_active=True,  gen=gen)
        self.opponent_active_pokemon  = MetamonPokemonProxy(opp_active_dict, is_active=True,  gen=gen)

        bench_proxies = [
            MetamonPokemonProxy(d, is_active=False, gen=gen) for d in bench_dicts
        ]

        # team dict: integer-keyed to preserve order; BattleStateGen1 calls list(.values())
        all_my = [self.active_pokemon] + bench_proxies
        self.team: dict[str, MetamonPokemonProxy] = {
            str(i): p for i, p in enumerate(all_my)
        }
        # Opponent bench slots 1–5 are intentionally absent: metamon replays only
        # record the active opponent.  TeamState zero-pads the missing slots, which
        # matches live-battle behavior where the bench is initially unseen.
        self.opponent_team: dict[str, MetamonPokemonProxy] = {
            '0': self.opponent_active_pokemon
        }

        # All of the active pokemon's known moves are "available" in metamon
        self.available_moves: list[Move] = list(self.active_pokemon.moves.values())

        # Arena fields consumed by ArenaStateGen1
        self.turn: int = turn
        self.side_conditions: dict          = _parse_side_conditions(
            state.get('player_conditions', [])
        )
        self.opponent_side_conditions: dict = _parse_side_conditions(
            state.get('opponent_conditions', [])
        )


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def metamon_to_obs_gen1(state: dict, turn: int) -> np.ndarray:
    """Convert a metamon state dict to a Gen-1 observation array.

    :param state: One element from ``d['states']`` in a metamon battle file.
    :param turn:  Zero-based turn index.
    :returns:     Float32 array of shape ``(BattleConfig.gen1().obs_dim,)``.
    """
    return BattleStateGen1(MetamonBattleProxy(state, turn)).to_array()
