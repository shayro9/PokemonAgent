/**
 * common.js
 * Shared helpers and config for data generation scripts.
 */
const { Teams } = require('../pokemon-showdown/dist/sim/teams');
const { Dex } = require('../pokemon-showdown/dist/sim/dex');
const { TeamValidator } = require('../pokemon-showdown/dist/sim/team-validator');

// ─── CONFIG ────────────────────────────────────────────────────────────────
const FORMAT = 'gen1ou';
const DEDUPE_EXACT_SETS = true;
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

function hasMirrorMove(mon) {
    return (mon.moves || []).some(m => Dex.moves.get(m).id === 'mirrormove');
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

module.exports = {
    Teams,
    FORMAT,
    DEDUPE_EXACT_SETS,
    setKey,
    canDamage,
    hasMirrorMove,
    pickOneMon,
    normalizeMon,
    isValidTeam,
};

