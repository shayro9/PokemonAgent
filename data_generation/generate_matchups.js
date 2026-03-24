/**
 * generate_matchups.js
 * Run: node generate_matchups.js
 */
const fs = require('fs');
const { Teams, FORMAT, DEDUPE_EXACT_SETS, setKey, canDamage, hasMirrorMove, pickOneMon, normalizeMon, isValidTeam } = require('./common');

// ─── CONFIG ────────────────────────────────────────────────────────────────
const OUTPUT_FILE     = '../data/matchups_1v1_gen1ou_db.json';
const NUM_TO_GENERATE = 10000;  // number of matchups to collect
const FULL_TEAMS      = false;  // true = 6-mon teams on each side, false = 1v1
// ───────────────────────────────────────────────────────────────────────────

function generateMatchupsPool() {
    const matchups = [];
    const seenKeys = new Set();
    let attempts = 0;
    let filteredUnwinnable = 0;

    console.log(`[matchups] Generating ${NUM_TO_GENERATE} winnable ${FULL_TEAMS ? '6v6' : '1v1'} matchups for ${FORMAT}...`);

    while (matchups.length < NUM_TO_GENERATE) {
        attempts++;

        const agentTeam = Teams.generate(FORMAT);
        const oppTeam   = Teams.generate(FORMAT);

        if (!isValidTeam(agentTeam) || !isValidTeam(oppTeam)) continue;

        const agentMons = FULL_TEAMS ? agentTeam : [pickOneMon(agentTeam)];
        const oppMons   = FULL_TEAMS ? oppTeam   : [pickOneMon(oppTeam)];

        // Winnability: every mon on each side must be able to damage at least one mon on the other
        const agentCanDamage = agentMons.every(a => oppMons.some(o => canDamage(a, o)));
        const oppCanDamage   = oppMons.every(o => agentMons.some(a => canDamage(o, a)));
        if (!agentCanDamage || !oppCanDamage) {
            filteredUnwinnable++;
            continue;
        }

        if ([...agentMons, ...oppMons].some(hasMirrorMove)) continue;

        if (DEDUPE_EXACT_SETS) {
            const key = `${agentMons.map(setKey).join('+')}__vs__${oppMons.map(setKey).join('+')}`;
            if (seenKeys.has(key)) continue;
            seenKeys.add(key);
        }

        matchups.push({
            agent:    agentMons.map(normalizeMon),
            opponent: oppMons.map(normalizeMon),
        });

        if (matchups.length % 1000 === 0)
            console.log(`  collected ${matchups.length} matchups (${attempts} attempts, ${filteredUnwinnable} unwinnable filtered)`);
    }

    console.log(`\nDone. ${filteredUnwinnable}/${attempts} matchups filtered as unwinnable (${(100 * filteredUnwinnable / attempts).toFixed(1)}%)`);

    return {
        format: FORMAT,
        mode: 'matchups',
        fullTeams: FULL_TEAMS,
        matchupCount: matchups.length,
        attempts,
        filteredUnwinnable,
        dedupeExactSets: DEDUPE_EXACT_SETS,
        pool: matchups,
    };
}

// ─── MAIN ──────────────────────────────────────────────────────────────────
try {
    const out = generateMatchupsPool();
    console.log(`\nWriting to ${OUTPUT_FILE}...`);
    fs.mkdirSync(require('path').dirname(OUTPUT_FILE), { recursive: true });
    fs.writeFileSync(OUTPUT_FILE, JSON.stringify(out, null, 2));
    console.log('Done!');
} catch (err) {
    console.error('Generation failed:', err);
    process.exit(1);
}

