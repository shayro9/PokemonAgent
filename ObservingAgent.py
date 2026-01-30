import asyncio

from poke_env.player.player import Player


class ObservingAgent(Player):
    def choose_move(self, battle):
        print(f"\n--- Turn {battle.turn} ---")
        print(f"Active Pokémons: {battle.active_pokemon.species} VS {battle.opponent_active_pokemon.species}")
        print(f"(HP: {battle.active_pokemon.current_hp_fraction * 100}%) Status: {battle.active_pokemon.status}")
        print(f"(HP: {battle.opponent_active_pokemon.current_hp_fraction * 100}%) Status: {battle.opponent_active_pokemon.status}")
        print(f"Available Moves: {[m.id for m in battle.available_moves]}")

        return self.choose_random_move(battle)
