import os

def log_fallback(self, battle) -> None:
    os.makedirs("data/fallback_logs", exist_ok=True)
    path = f"data/fallback_logs/{battle.battle_tag.replace('|', '_')}_turn{battle.turn}.log"

    with open(path, "w") as f:
        for msg in battle.observations:  # raw Showdown protocol messages
            f.write(msg + "\n")

    print(f"[FALLBACK LOG] {path}")