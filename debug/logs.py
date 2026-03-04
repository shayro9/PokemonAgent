import os

def log_fallback(battle) -> None:
    os.makedirs("data/fallback_logs", exist_ok=True)
    path = f"data/fallback_logs/{battle.battle_tag.replace('|', '_')}_turn{battle.turn}.log"

    with open(path, "w") as f:
        for msg in battle.observations:
            f.write(str(msg) + "\n")

    print(f"[FALLBACK LOG] {path}")