from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from env.embed import (
    embed_move,
    embed_status,
    embed_effects,
    embed_weather,
    calc_types_vector,
    MAX_MOVES,
    MOVE_EMBED_LEN,
    MOVE_STATUSES, TRACKED_EFFECTS,
)

# Update this constant whenever embed_battle changes.
# opp_stat_belief shrank from 12 to 10 (HP removed from belief).
OBS_SIZE = 373
CONTEXT_BEFORE_MY_MOVES = 68
CONTEXT_AFTER_OPP_MOVES = 1


@dataclass
class BattleState:
    """
    Structured representation of a single battle observation.
    Call .to_array() to get the flat np.ndarray passed to the model.
    """
    turn: float                    # (1)
    weather: np.ndarray            # (9)

    my_hp: float                   # (1)
    my_stats: np.ndarray           # (6),  base stats normalised
    my_boosts: np.ndarray          # (7)  in-battle boosts normalised
    my_status: np.ndarray          # (7)  one-hot Status
    my_effects: np.ndarray         # (3)  one-hot Tracked effects
    my_stab: float                 # (1)

    opp_hp: float                  # (1)
    opp_stat_belief: np.ndarray    # (10,)  [mean/STAT_NORM ×5, std/STAT_NORM ×5]  (no HP)
    opp_boosts: np.ndarray         # (7,)
    opp_status: np.ndarray         # (7,)
    opp_effects: np.ndarray        # (3)  one-hot Tracked effects
    opp_preparing: float           # 0 or 1
    opp_stab: float                # (1)
    opp_is_terastallized: float    # 0 or 1
    opp_tera_multiplier: np.ndarray# (2)

    my_moves: np.ndarray           # (MAX_MOVES * MOVE_EMBED_LEN,)
    opp_moves: np.ndarray          # (MAX_MOVES * MOVE_EMBED_LEN,)

    opp_protect_belief: float      # (1)

    def to_array(self) -> np.ndarray:
        state = np.concatenate([
            [self.turn],
            self.weather,
            [self.my_hp],
            self.my_stats,
            self.my_boosts,
            self.my_status,
            self.my_effects,
            [self.my_stab],
            [self.opp_hp],
            self.opp_stat_belief,
            self.opp_boosts,
            self.opp_status,
            self.opp_effects,
            [self.opp_preparing, self.opp_stab, self.opp_is_terastallized],
            self.opp_tera_multiplier,
            self.my_moves,
            self.opp_moves,
            [self.opp_protect_belief],
        ]).astype(np.float32)

        assert len(state) == OBS_SIZE, (
            f"BattleState.to_array(): expected {OBS_SIZE} dims, got {len(state)}. "
            "Update OBS_SIZE if you added new features."
        )
        return state

    @classmethod
    def from_battle(cls, battle, opp_protect_belief: float = 1.0, opp_stat_belief: np.ndarray | None = None) -> "BattleState":
        """Build a BattleState from a poke-env battle object."""
        my = battle.active_pokemon
        opp = battle.opponent_active_pokemon

        my_types = my.types
        opp_types = opp.types

        # --- moves ---
        my_moves = np.zeros(MAX_MOVES * MOVE_EMBED_LEN, dtype=np.float32)
        for i, move in enumerate(battle.available_moves[:MAX_MOVES]):
            emb = embed_move(move, opp_types, my_types, battle.gen)
            my_moves[i * MOVE_EMBED_LEN: (i + 1) * MOVE_EMBED_LEN] = emb

        opp_moves = np.zeros(MAX_MOVES * MOVE_EMBED_LEN, dtype=np.float32)
        opp_moves_list = list((opp.moves or {}).values())[:MAX_MOVES]
        for i, move in enumerate(opp_moves_list):
            emb = embed_move(move, my_types, opp_types, battle.gen)
            opp_moves[i * MOVE_EMBED_LEN: (i + 1) * MOVE_EMBED_LEN] = emb

        return cls(
            turn=battle.turn / 30.0,
            weather=embed_weather(battle.weather),
            my_hp=my.current_hp_fraction,
            my_stats=np.minimum(np.array(list(my.stats.values())) / 512.0, 1.0),
            my_boosts=np.array(list(my.boosts.values())) / 6.0,
            my_status=embed_status(my.status),
            my_effects=embed_effects(my.effects),
            my_stab=my.stab_multiplier / 2.0,

            opp_hp=opp.current_hp_fraction,
            opp_stat_belief=opp_stat_belief if opp_stat_belief is not None else np.zeros(10, dtype=np.float32),
            opp_boosts=np.array(list(opp.boosts.values())) / 6.0,
            opp_status=embed_status(opp.status),
            opp_effects=embed_effects(opp.effects),
            opp_preparing=float(opp.preparing),
            opp_stab=opp.stab_multiplier / 2.0,
            opp_is_terastallized=float(opp.is_terastallized),
            opp_tera_multiplier=calc_types_vector(my_types, [opp.tera_type], battle.gen, opp_tera_mode=True),

            my_moves=my_moves,
            opp_moves=opp_moves,
            opp_protect_belief=opp_protect_belief,
        )

    def describe(self) -> str:
        """Human-readable breakdown matching to_array() layout. Useful for debugging."""
        arr = self.to_array()
        layout = [
            ("my_hp", 1),
            ("my_stats", 6),
            ("my_boosts", 7),
            ("my_status", len(MOVE_STATUSES)),
            ("my_effects", len(TRACKED_EFFECTS)),
            ("my_stab", 1),
            ("opp_hp", 1),
            ("opp_stats_belief", 10),
            ("opp_boosts", 7),
            ("opp_status", len(MOVE_STATUSES)),
            ("opp_effects", len(TRACKED_EFFECTS)),
            ("opp_preparing", 1),
            ("opp_stab", 1),
            ("opp_is_terastallized", 1),
            ("opp_tera_multiplier", 1),
        ]
        move_block = [
            ("scalars", 5),
            ("category", len(list(__import__('poke_env.battle.move_category', fromlist=['MoveCategory']).MoveCategory))),
            ("is_protect", 1),
            ("breaks_protect", 1),
            ("multiplier", 1),
            ("is_tera", 1),
            ("status", len(MOVE_STATUSES)),
            ("boosts_target", 7),
            ("boosts_self", 7),
            ("recoil_drain", 2),
            ("multi_hit", 2),
        ]
        for i in range(1, MAX_MOVES + 1):
            layout.extend([(f"move{i}.{name}", size) for name, size in move_block])

        for i in range(1, MAX_MOVES + 1):
            layout.extend([(f"opp_move{i}.{name}", size) for name, size in move_block])

        lines = ["[BattleState] OBSERVATION BREAKDOWN"]
        idx = 0
        for name, size in layout:
            chunk = arr[idx: idx + size]
            if size == 1:
                lines.append(f"  {name:35}: {chunk[0]: .3f}")
            else:
                fmt = ", ".join(f"{x: .3f}" for x in chunk)
                lines.append(f"  {name:35}: [{fmt}]")
            idx += size

        lines.append(f"\n  TOTAL DIMENSIONS: {len(arr)}")
        return "\n".join(lines)
