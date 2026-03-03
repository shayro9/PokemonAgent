"""
count_pokemon.py
Usage: python count_pokemon.py <path_to_db.json>
"""
import json
import sys
from collections import Counter

path = sys.argv[1] if len(sys.argv) > 1 else '../data/matchups_gen9randombattle_db.json'

with open(path) as f:
    db = json.load(f)

mode = db.get('mode', 'unknown')
pool = db.get('pool', [])

counter = Counter()

if mode == 'matchups':
    for entry in pool:
        counter[entry['agent']['species']] += 1
elif mode == 'teams':
    for mon in pool:
        counter[mon['species']] += 1
else:
    print(f"Unknown mode: {mode}")
    sys.exit(1)

print(f"Mode        : {mode}")
print(f"Total entries: {len(pool)}")
print(f"Unique Pokémon: {len(counter)}\n")
print(f"{'Species':<25} {'Count':>6}")
print("-" * 33)
for species, count in sorted(counter.items(), key=lambda x: -x[1]):
    print(f"{species:<25} {count:>6}")