/**
 * generate_gen1ou_db.js
 * Run: node generate_gen1ou_db.js
 */
const fs = require('fs');
const { Teams } = require('../pokemon-showdown/dist/sim/teams.js');
const { Dex } = require('../pokemon-showdown/dist/sim/dex.js');

// ─── CONFIG ────────────────────────────────────────────────────────────────
const FORMAT = 'gen1ou';
const OUTPUT_FILE = './data/gen1ou_teams_db.json';

// 'teams' → original behaviour (one mon per entry)
const MODE = 'teams';

const NUM_TO_GENERATE = 10000;   // teams to attempt
const MAX_PER_SPECIES  = 60;
const DEDUPE_EXACT_SETS = true;
const ONE_MON_PER_TEAM  = true;
const FILTER_SPECIES    = null;  // e.g. 'Tauros', or null for no filter
// ───────────────────────────────────────────────────────────────────────────

// ─── HELPERS ───────────────────────────────────────────────────────────────
function setKey(s) {
    const moves = (s.moves || []).slice().sort().join(',');
    const evs = s.evs ? JSON.stringify(s.evs) : '';
    const ivs = s.ivs ? JSON.stringify(s.ivs) : '';
    return [s.species, s.item, s.ability, moves, s.nature, evs, ivs,
            s.gender, s.level, s.shiny, s.teraType].join('|');
}

function pickOneMon(team) {
    return team[Math.floor(Math.random() * team.length)];
}
// ───────────────────────────────────────────────────────────────────────────

// ─── TEAMS MODE ────────────────────────────────────────────────────────────
function generateTeamsPool() {
    const pool = [];
    const perSpeciesCount = new Map();
    const seenSetKeys = new Set();

    console.log(`[teams] Generating up to ${NUM_TO_GENERATE} teams for ${FORMAT}...`);

    for (let i = 0; i < NUM_TO_GENERATE; i++) {
        const team = Teams.generate(FORMAT);
        const picks = ONE_MON_PER_TEAM ? [pickOneMon(team)] : team;

        for (const mon of picks) {
            const sp = mon.species;
            if (FILTER_SPECIES && sp !== FILTER_SPECIES) continue;

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

        if ((i + 1) % 1000 === 0)
            console.log(`  attempted ${i + 1} teams, pool size: ${pool.length}`);
    }

    return {
        format: FORMAT,
        mode: 'teams',
        generatedTeams: NUM_TO_GENERATE,
        oneMonPerTeam: ONE_MON_PER_TEAM,
        maxPerSpecies: MAX_PER_SPECIES,
        dedupeExactSets: DEDUPE_EXACT_SETS,
        poolSize: pool.length,
        pool,
    };
}
// ───────────────────────────────────────────────────────────────────────────

// ─── MAIN ──────────────────────────────────────────────────────────────────
try {
    const out = generateTeamsPool();
    console.log(`\nWriting to ${OUTPUT_FILE}...`);
    fs.writeFileSync(OUTPUT_FILE, JSON.stringify(out, null, 2));
    console.log('Done!');
} catch (err) {
    console.error('Generation failed:', err);
    process.exit(1);
}
