/**
 * generate_teams.js
 * Run: node generate_teams.js
 */
const fs = require('fs');
const { Teams, FORMAT, DEDUPE_EXACT_SETS, setKey, hasMirrorMove, pickOneMon, normalizeMon, isValidTeam } = require('./common');

// ─── CONFIG ────────────────────────────────────────────────────────────────
const OUTPUT_FILE    = './data/teams_gen1ou_db.json';
const NUM_TO_GENERATE = 10000;  // number of teams to attempt
const MAX_PER_SPECIES = 60;
const ONE_MON_PER_TEAM = true;
const FILTER_SPECIES   = null;  // e.g. 'Starmie', or null for no filter
// ───────────────────────────────────────────────────────────────────────────

function generateTeamsPool() {
    const pool = [];
    const perSpeciesCount = new Map();
    const seenSetKeys = new Set();

    console.log(`[teams] Generating up to ${NUM_TO_GENERATE} teams for ${FORMAT}...`);

    for (let i = 0; i < NUM_TO_GENERATE; i++) {
        const team = Teams.generate(FORMAT);
        if (!isValidTeam(team)) continue;
        const picks = ONE_MON_PER_TEAM ? [pickOneMon(team)] : team;

        for (const mon of picks) {
            const sp = mon.species;
            if (FILTER_SPECIES && sp !== FILTER_SPECIES) continue;
            if (hasMirrorMove(mon)) continue;

            const cnt = perSpeciesCount.get(sp) || 0;
            if (cnt >= MAX_PER_SPECIES) continue;

            if (DEDUPE_EXACT_SETS) {
                const key = setKey(mon);
                if (seenSetKeys.has(key)) continue;
                seenSetKeys.add(key);
            }

            perSpeciesCount.set(sp, cnt + 1);
            pool.push(normalizeMon(mon));
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

// ─── MAIN ──────────────────────────────────────────────────────────────────
try {
    const out = generateTeamsPool();
    console.log(`\nWriting to ${OUTPUT_FILE}...`);
    fs.mkdirSync(require('path').dirname(OUTPUT_FILE), { recursive: true });
    fs.writeFileSync(OUTPUT_FILE, JSON.stringify(out, null, 2));
    console.log('Done!');
} catch (err) {
    console.error('Generation failed:', err);
    process.exit(1);
}


