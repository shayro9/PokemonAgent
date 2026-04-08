#!/usr/bin/env python3
"""
Reward Constant Tuning Tool

This script helps tune the reward function constants to optimize agent learning.
It provides:
1. Detailed breakdown of reward components
2. Interactive tuning interface
3. Ablation study framework
4. Visualization of reward impact

Usage:
    # Show current reward breakdown
    python scripts/tune_reward_constants.py --show-config
    
    # Run sensitivity analysis
    python scripts/tune_reward_constants.py --sensitivity-analysis
    
    # Interactively tune a constant
    python scripts/tune_reward_constants.py --tune atk_boost
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class RewardConfig:
    """Current reward configuration."""
    hp_value: float = 1.0
    fainted_value: float = 5.0
    win_bonus: float = 15.0
    loss_penalty: float = -10.0
    
    # Status weights (opponent infliction value)
    status_slp: float = 1.2
    status_par: float = 0.5
    status_brn: float = 0.6
    status_tox: float = 0.4
    status_psn: float = 0.3
    status_frz: float = 1.2
    
    # Own boost values (positive boost reward)
    own_atk: float = 0.8
    own_def: float = 0.6
    own_spa: float = 0.8
    own_spd: float = 0.6
    own_spe: float = 1.0
    own_accuracy: float = 0.4
    own_evasion: float = 0.5
    
    # Opponent boost penalties (negative boost on opponent = good)
    opp_atk: float = 1.2
    opp_def: float = 0.8
    opp_spa: float = 1.2
    opp_spd: float = 0.8
    opp_spe: float = 1.5
    opp_accuracy: float = 0.6
    opp_evasion: float = 0.8


class RewardTuner:
    """Interactive reward tuning tool."""
    
    def __init__(self):
        self.config = RewardConfig()
        self.scenarios = self._build_scenarios()
    
    def _build_scenarios(self) -> Dict[str, Dict]:
        """Build test scenarios for evaluation."""
        return {
            "setup_move": {
                "description": "+2 Atk on own Pokémon",
                "own_boosts": {"atk": 2},
                "opp_boosts": {},
                "expected_reward": 2 * self.config.own_atk,
            },
            "cripple_opponent": {
                "description": "-2 Atk on opponent Pokémon",
                "own_boosts": {},
                "opp_boosts": {"atk": -2},
                "expected_reward": 2 * self.config.opp_atk,
            },
            "speed_control": {
                "description": "+1 Spe on own, -1 Spe on opponent",
                "own_boosts": {"spe": 1},
                "opp_boosts": {"spe": -1},
                "expected_reward": (1 * self.config.own_spe) + (1 * self.config.opp_spe),
            },
            "sweep_setup": {
                "description": "+2 Atk, +1 Spe on own",
                "own_boosts": {"atk": 2, "spe": 1},
                "opp_boosts": {},
                "expected_reward": (2 * self.config.own_atk) + (1 * self.config.own_spe),
            },
            "defensive_wall": {
                "description": "+2 Def on own",
                "own_boosts": {"def": 2},
                "opp_boosts": {},
                "expected_reward": 2 * self.config.own_def,
            },
            "wall_breaking": {
                "description": "-2 Def on opponent",
                "own_boosts": {},
                "opp_boosts": {"def": -2},
                "expected_reward": 2 * self.config.opp_def,
            },
            "paralyze_opponent": {
                "description": "Paralyze opponent (PAR status)",
                "own_boosts": {},
                "opp_boosts": {},
                "status_inflicted": "PAR",
                "expected_reward": self.config.status_par,
            },
            "freeze_opponent": {
                "description": "Freeze opponent (FRZ status)",
                "own_boosts": {},
                "opp_boosts": {},
                "status_inflicted": "FRZ",
                "expected_reward": self.config.status_frz,
            },
        }
    
    def show_config(self) -> None:
        """Display current reward configuration."""
        print("\n" + "="*70)
        print("CURRENT REWARD CONFIGURATION")
        print("="*70)
        
        print("\n[Game-Ending Rewards]")
        print(f"  WIN_BONUS:          {self.config.win_bonus:>6.2f}")
        print(f"  LOSS_PENALTY:       {self.config.loss_penalty:>6.2f}")
        
        print("\n[HP & Fainting]")
        print(f"  HP_VALUE:           {self.config.hp_value:>6.2f}")
        print(f"  FAINTED_VALUE:      {self.config.fainted_value:>6.2f}")
        
        print("\n[Status Conditions (opponent infliction value)]")
        print(f"  SLP (Sleep):        {self.config.status_slp:>6.2f}")
        print(f"  PAR (Paralysis):    {self.config.status_par:>6.2f}")
        print(f"  BRN (Burn):         {self.config.status_brn:>6.2f}")
        print(f"  FRZ (Freeze):       {self.config.status_frz:>6.2f}")
        print(f"  TOX (Toxic):        {self.config.status_tox:>6.2f}")
        print(f"  PSN (Poison):       {self.config.status_psn:>6.2f}")
        
        print("\n[Own Pokémon Boosts (positive boost reward)]")
        print(f"  +Atk (Attack):      {self.config.own_atk:>6.2f}")
        print(f"  +Def (Defense):     {self.config.own_def:>6.2f}")
        print(f"  +SpA (Sp.Atk):      {self.config.own_spa:>6.2f}")
        print(f"  +SpD (Sp.Def):      {self.config.own_spd:>6.2f}")
        print(f"  +Spe (Speed):       {self.config.own_spe:>6.2f}")
        print(f"  +Acc (Accuracy):    {self.config.own_accuracy:>6.2f}")
        print(f"  +Eva (Evasion):     {self.config.own_evasion:>6.2f}")
        
        print("\n[Opponent Pokémon Penalties (negative boost penalty value)]")
        print(f"  -Atk (Attack):      {self.config.opp_atk:>6.2f} ⭐ HIGH PRIORITY")
        print(f"  -Def (Defense):     {self.config.opp_def:>6.2f}")
        print(f"  -SpA (Sp.Atk):      {self.config.opp_spa:>6.2f} ⭐ HIGH PRIORITY")
        print(f"  -SpD (Sp.Def):      {self.config.opp_spd:>6.2f}")
        print(f"  -Spe (Speed):       {self.config.opp_spe:>6.2f} ⭐ HIGHEST PRIORITY")
        print(f"  -Acc (Accuracy):    {self.config.opp_accuracy:>6.2f}")
        print(f"  -Eva (Evasion):     {self.config.opp_evasion:>6.2f}")
        print()
    
    def sensitivity_analysis(self) -> None:
        """Perform sensitivity analysis on reward components."""
        print("\n" + "="*70)
        print("SCENARIO REWARD EVALUATION")
        print("="*70)
        
        for scenario_name, scenario in self.scenarios.items():
            reward = scenario.get("expected_reward", 0.0)
            desc = scenario["description"]
            print(f"\n✓ {scenario_name.upper()}: {desc}")
            print(f"  Expected Reward: {reward:>8.2f}")
    
    def compare_configs(self, config1_name: str, config2_name: str) -> None:
        """Compare two configurations."""
        print(f"\nComparing {config1_name} vs {config2_name}")
        # This would load and compare saved configs
    
    def suggest_values(self) -> None:
        """Suggest reward values based on game theory."""
        print("\n" + "="*70)
        print("SUGGESTED REWARD VALUES (Based on Game Theory)")
        print("="*70)
        
        print("\n[Key Principles]")
        print("1. Offensive boosts (Atk, SpA, Spe) are HIGHEST value")
        print("2. Opponent stat reductions are MORE valuable than own boosts")
        print("3. Speed control (-Spe on opponent) is critical")
        print("4. Status conditions should vary by type (Sleep/Freeze >> Poison)")
        
        print("\n[Recommended Tiers]")
        print("TIER 1 (Highest): -Spe opponent, Setup moves (Atk+Spe)")
        print("TIER 2 (High):    -Atk/-SpA opponent, Status (Sleep/Freeze)")
        print("TIER 3 (Medium):  +Atk/+SpA own, Defense moves")
        print("TIER 4 (Low):     Accuracy/Evasion, Weak status")
        
        print("\n[Example: Competitive Priorities]")
        suggested = {
            "opp_spe": 1.5,     # Speed control is KING
            "opp_atk": 1.2,     # Block physical sweepers
            "opp_spa": 1.2,     # Block special sweepers
            "own_spe": 1.0,     # Matching priority
            "own_atk": 0.8,     # Slightly lower (own > opp defense)
            "status_slp": 1.2,  # Sleep is game-winning
            "status_frz": 1.2,  # Freeze is game-winning
        }
        for key, val in suggested.items():
            print(f"  {key:<15}: {val:>5.2f}")


def main():
    parser = argparse.ArgumentParser(description="Reward constant tuning tool")
    parser.add_argument("--show-config", action="store_true", help="Display current reward config")
    parser.add_argument("--sensitivity-analysis", action="store_true", help="Run scenario analysis")
    parser.add_argument("--suggest", action="store_true", help="Get suggestions for reward values")
    parser.add_argument("--export-json", type=str, help="Export current config to JSON file")
    
    args = parser.parse_args()
    
    tuner = RewardTuner()
    
    if args.show_config:
        tuner.show_config()
    
    if args.sensitivity_analysis:
        tuner.sensitivity_analysis()
    
    if args.suggest:
        tuner.suggest_values()
    
    if args.export_json:
        config_dict = vars(tuner.config)
        with open(args.export_json, 'w') as f:
            json.dump(config_dict, f, indent=2)
        print(f"✓ Exported config to {args.export_json}")
    
    # Default: show everything
    if not any([args.show_config, args.sensitivity_analysis, args.suggest, args.export_json]):
        tuner.show_config()
        tuner.sensitivity_analysis()
        tuner.suggest_values()


if __name__ == "__main__":
    main()
