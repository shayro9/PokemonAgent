/**
 * generate_db.js
 * Run: node generate_db.js
 */
const fs = require('fs');
const { Teams } = require('../pokemon-showdown/dist/sim/teams');
const { Dex } = require('../pokemon-showdown/dist/sim/dex');
const { TeamValidator } = require('../pokemon-showdown/dist/sim/team-validator');

// ─── CONFIG ────────────────────────────────────────────────────────────────
const FORMAT = 'gen1ou';
const OUTPUT_FILE = './data/matchups_gen1ou_db.json';

// 'teams' → original behaviour (one mon per entry)
// 'matchups' → paired 1v1 matchups with winnability filter
const MODE = 'matchups';

const NUM_TO_GENERATE = 10000;   // teams to attempt (teams mode) or matchups to collect (matchups mode)
const MAX_PER_SPECIES  = 60;     // ignored in matchups mode
const DEDUPE_EXACT_SETS = true;
const ONE_MON_PER_TEAM  = true;  // teams mode only
const FILTER_SPECIES    = null;  // e.g. 'Bellibolt', or null for no filter
// ───────────────────────────────────────────────────────────────────────────

// ─── HELPERS ───────────────────────────────────────────────────────────────
function setKey(s) {
    const moves = (s.moves || []).slice().sort().join(',');
    const evs = s.evs ? JSON.stringify(s.evs) : '';
    const ivs = s.ivs ? JSON.stringify(s.ivs) : '';
    return [s.species, s.item, s.ability, moves, s.nature, evs, ivs,
            s.gender, s.level, s.shiny, s.teraType].join('|');
}

function getTypes(mon) {
    const species = Dex.species.get(mon.species);
    return species.exists ? species.types : ['Normal'];
}

function getMoveTypes(mon) {
    return (mon.moves || []).map(moveName => {
        const move = Dex.moves.get(moveName);
        return move.exists ? move.type : 'Normal';
    });
}

function canDamage(attacker, defender) {
    const defTypes = getTypes(defender);
    return getMoveTypes(attacker).some(atkType => {
        const mult = defTypes.reduce((acc, defType) => {
            const chart = Dex.types.get(atkType);
            const val = chart.exists ? (chart.damageTaken[defType] ?? 0) : 0;
            // damageTaken: 0 = normal, 1 = immune, 2 = resist, 3 = weak
            if (val === 1) return 0;       // immune
            if (val === 2) return acc * 0.5;
            if (val === 3) return acc * 2;
            return acc;
        }, 1);
        return mult > 0;
    });
}

function pickOneMon(team) {
    return team[Math.floor(Math.random() * team.length)];
}

function normalizeMon(mon) {
    // Teams.generate() sets gender=false for genderless mons (Gen 1).
    // poke-env can't handle the stringified 'False' — coerce to '' instead.
    if (!mon.gender) mon.gender = '';
    return mon;
}

const _validator = new TeamValidator(FORMAT);

/** Returns true only if the full team passes Showdown's server-side rules. */
function isValidTeam(team) {
    const problems = _validator.validateTeam(team);
    return !problems || problems.length === 0;
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
        if (!isValidTeam(team)) continue;
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
// ───────────────────────────────────────────────────────────────────────────

// ─── MATCHUPS MODE ─────────────────────────────────────────────────────────
function generateMatchupsPool() {
    const matchups = [];
    const seenKeys = new Set();
    let attempts = 0;
    let filteredUnwinnable = 0;

    console.log(`[matchups] Generating ${NUM_TO_GENERATE} winnable 1v1 matchups for ${FORMAT}...`);

    while (matchups.length < NUM_TO_GENERATE) {
        attempts++;

        const agentTeam = Teams.generate(FORMAT);
        const oppTeam   = Teams.generate(FORMAT);

        if (!isValidTeam(agentTeam) || !isValidTeam(oppTeam)) continue;

        const agent = pickOneMon(agentTeam);
        const opp   = pickOneMon(oppTeam);

        // Winnability: both sides must be able to deal damage
        if (!canDamage(agent, opp) || !canDamage(opp, agent)) {
            filteredUnwinnable++;
            continue;
        }

        if (DEDUPE_EXACT_SETS) {
            const key = `${setKey(agent)}__vs__${setKey(opp)}`;
            if (seenKeys.has(key)) continue;
            seenKeys.add(key);
        }

        matchups.push({ agent: normalizeMon(agent), opponent: normalizeMon(opp) });

        if (matchups.length % 1000 === 0)
            console.log(`  collected ${matchups.length} matchups (${attempts} attempts, ${filteredUnwinnable} unwinnable filtered)`);
    }

    console.log(`\nDone. ${filteredUnwinnable}/${attempts} matchups filtered as unwinnable (${(100 * filteredUnwinnable / attempts).toFixed(1)}%)`);

    return {
        format: FORMAT,
        mode: 'matchups',
        matchupCount: matchups.length,
        attempts,
        filteredUnwinnable,
        dedupeExactSets: DEDUPE_EXACT_SETS,
        pool: matchups,
    };
}
// ───────────────────────────────────────────────────────────────────────────

// ─── MAIN ──────────────────────────────────────────────────────────────────
try {
    const out = MODE === 'matchups' ? generateMatchupsPool() : generateTeamsPool();
    console.log(`\nWriting to ${OUTPUT_FILE}...`);
    fs.writeFileSync(OUTPUT_FILE, JSON.stringify(out, null, 2));
    console.log('Done!');
} catch (err) {
    console.error('Generation failed:', err);
    process.exit(1);
}