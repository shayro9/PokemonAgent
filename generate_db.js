/**
 * generate_db.js
 * Run: node generate_db.js
 */
const fs = require('fs');
const { Teams } = require('./pokemon-showdown/dist/sim/teams');

// CONFIG
const FORMAT = 'gen9randombattle';
const NUM_TEAMS_TO_GENERATE = 20000;
const OUTPUT_FILE = './data/gen9randombattle_db.json';

// Optional knobs
const ONE_MON_PER_TEAM = true;        // Step-1 recommendation
const MAX_PER_SPECIES = 60;           // prevent a few species dominating the DB
const DEDUPE_EXACT_SETS = true;       // remove identical sets

// Helpers
function setKey(s) {
  // create a stable key so identical sets dedupe
  const moves = (s.moves || []).slice().sort().join(',');
  const evs = s.evs ? JSON.stringify(s.evs) : '';
  const ivs = s.ivs ? JSON.stringify(s.ivs) : '';
  return [
    s.species, s.item, s.ability, moves, s.nature, evs, ivs, s.gender, s.level, s.shiny, s.teraType
  ].join('|');
}

const pool = [];
const perSpeciesCount = new Map();
const seenSetKeys = new Set();

console.log(`Generating ${NUM_TEAMS_TO_GENERATE} teams for ${FORMAT}...`);

try {
  for (let i = 0; i < NUM_TEAMS_TO_GENERATE; i++) {
    const team = Teams.generate(FORMAT); // array of PokemonSet objects

    const picks = ONE_MON_PER_TEAM
      ? [team[Math.floor(Math.random() * team.length)]]
      : team;

    for (const mon of picks) {
      const sp = mon.species;
      const cnt = perSpeciesCount.get(sp) || 0;
      if (cnt >= MAX_PER_SPECIES) continue;

      if (DEDUPE_EXACT_SETS) {
        const key = setKey(mon);
        if (seenSetKeys.has(key)) continue;
        seenSetKeys.add(key);
      }

      perSpeciesCount.set(sp, cnt + 1);
      pool.push(mon);
    }

    if ((i + 1) % 1000 === 0) {
      console.log(`Generated ${i + 1} teams. Pool size so far: ${pool.length}`);
    }
  }

  const out = {
    format: FORMAT,
    generatedTeams: NUM_TEAMS_TO_GENERATE,
    oneMonPerTeam: ONE_MON_PER_TEAM,
    maxPerSpecies: MAX_PER_SPECIES,
    dedupeExactSets: DEDUPE_EXACT_SETS,
    poolSize: pool.length,
    pool,
  };

  console.log(`Writing ${pool.length} mons to ${OUTPUT_FILE}`);
  fs.writeFileSync(OUTPUT_FILE, JSON.stringify(out, null, 2));
  console.log('Done!');
} catch (err) {
  console.error('Generation failed:', err);
}
