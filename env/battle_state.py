from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from env.embed import (
    embed_move,
    embed_status,
    embed_effects,
    calc_types_vector,
    MAX_MOVES,
    MOVE_EMBED_LEN,
    MOVE_STATUSES, TRACKED_EFFECTS,
)

# Update this constant whenever embed_battle changes.
OBS_SIZE = 341  # prior features + protect belief posterior and expected next chance


@dataclass
class BattleState:
    """
    Structured representation of a single battle observation.
    Call .to_array() to get the flat np.ndarray passed to the model.
    """
    my_hp: float
    my_stats: np.ndarray           # (6),  base stats normalised
    my_boosts: np.ndarray          # (7)  in-battle boosts normalised
    my_status: np.ndarray          # (7)  one-hot Status
    my_effects: np.ndarray         # (3)  one-hot Tracked effects

    opp_hp: float
    opp_base_stats: np.ndarray     # (6,)
    opp_boosts: np.ndarray         # (7,)
    opp_status: np.ndarray         # (7,)
    opp_effects: np.ndarray        # (3)  one-hot Tracked effects
    opp_preparing: float           # 0 or 1
    protect_posterior: float       # P(protect_success | no_damage)
    protect_next_chance: float     # E[next protect success chance]

    my_moves: np.ndarray           # (MAX_MOVES * MOVE_EMBED_LEN,)
    opp_moves: np.ndarray          # (MAX_MOVES * MOVE_EMBED_LEN,)

    type_multipliers: np.ndarray   # (4,)
    weight_bucket: np.ndarray      # (6,)  one-hot

    def to_array(self) -> np.ndarray:
        state = np.concatenate([
            [self.my_hp],
            self.my_stats,
            self.my_boosts,
            self.my_status,
            self.my_effects,
            [self.opp_hp],
            self.opp_base_stats,
            self.opp_boosts,
            self.opp_status,
            self.opp_effects,
            [self.opp_preparing],
            [self.protect_posterior],
            [self.protect_next_chance],
            self.my_moves,
            self.opp_moves,
            self.type_multipliers,
            self.weight_bucket,
        ]).astype(np.float32)

        assert len(state) == OBS_SIZE, (
            f"BattleState.to_array(): expected {OBS_SIZE} dims, got {len(state)}. "
            "Update OBS_SIZE if you added new features."
        )
        return state

    @classmethod
    def from_battle(
        cls,
        battle,
        *,
        protect_posterior: float = 0.0,
        protect_next_chance: float = 1.0,
    ) -> "BattleState":
        """Build a BattleState from a poke-env battle object."""
        my = battle.active_pokemon
        opp = battle.opponent_active_pokemon

        my_types = my.types
        opp_types = opp.types

        # --- moves ---
        my_moves = np.zeros(MAX_MOVES * MOVE_EMBED_LEN, dtype=np.float32)
        for i, move in enumerate(battle.available_moves[:MAX_MOVES]):
            emb = embed_move(move, opp_types, battle.gen)
            my_moves[i * MOVE_EMBED_LEN: (i + 1) * MOVE_EMBED_LEN] = emb

        opp_moves = np.zeros(MAX_MOVES * MOVE_EMBED_LEN, dtype=np.float32)
        opp_moves_list = list((opp.moves or {}).values())[:MAX_MOVES]
        for i, move in enumerate(opp_moves_list):
            emb = embed_move(move, my_types, battle.gen)
            opp_moves[i * MOVE_EMBED_LEN: (i + 1) * MOVE_EMBED_LEN] = emb

        # --- weight bucket ---
        bucket = int(my.weight // opp.weight) if opp.weight else 0
        bucket = max(0, min(bucket, 5))
        weight_bucket = np.zeros(6, dtype=np.float32)
        weight_bucket[bucket] = 1.0

        return cls(
            my_hp=my.current_hp_fraction,
            my_stats=np.minimum(np.array(list(my.stats.values())) / 255.0, 1.0),
            my_boosts=np.array(list(my.boosts.values())) / 6.0,
            my_status=embed_status(my.status),
            my_effects=embed_effects(my.effects),

            opp_hp=opp.current_hp_fraction,
            opp_base_stats=np.array(list(opp.base_stats.values())) / 255.0,
            opp_boosts=np.array(list(opp.boosts.values())) / 6.0,
            opp_status=embed_status(opp.status),
            opp_effects=embed_effects(opp.effects),
            opp_preparing=float(opp.preparing),
            protect_posterior=float(protect_posterior),
            protect_next_chance=float(protect_next_chance),

            my_moves=my_moves,
            opp_moves=opp_moves,

            type_multipliers=calc_types_vector(my_types, opp_types, battle.gen),
            weight_bucket=weight_bucket,
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
            ("opp_hp", 1),
            ("opp_base_stats", 6),
            ("opp_boosts", 7),
            ("opp_status", len(MOVE_STATUSES)),
            ("opp_effects", len(TRACKED_EFFECTS)),
            ("opp_preparing", 1),
            ("protect_posterior", 1),
            ("protect_next_chance", 1),
        ]
        move_block = [
            ("scalars", 4),
            ("category", len(list(__import__('poke_env.battle.move_category', fromlist=['MoveCategory']).MoveCategory))),
            ("is_protect", 1),
            ("breaks_protect", 1),
            ("multiplier", 1),
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
        layout += [("type_multipliers", 4), ("weight_bucket", 6)]

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
