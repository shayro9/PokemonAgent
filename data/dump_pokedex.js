const { Dex } = require('../pokemon-showdown/dist/sim/dex');
const { TeamValidator } = require('../pokemon-showdown/dist/sim/team-validator');
const fs = require('fs');

const formatId = 'gen9nationaldex';
const dex = Dex.forFormat(formatId);
const validator = new TeamValidator(formatId);
const output = {};

// Access the format rules directly (this pulls from formats.ts logic)
const formatRules = dex.formats.get(formatId);
const banlist = formatRules.banlist || [];

console.log(`Analyzing ${formatId} using official banlists...`);

for (const species of dex.species.all()) {
    // 1. Tier & Standard Filtering (Uses tags logic)
    if (!species.exists || species.isNonstandard === 'Custom' || species.isNonstandard === 'CAP') continue;

    // Check if the species itself is in the format's specific banlist
    if (banlist.includes(species.name)) continue;

    // Filter out Ubers and Unreleased (Darkrai fix)
    const tier = species.tier;
    if (tier.includes('Uber') || tier === 'AG' || tier === 'Illegal' || tier === 'Unreleased') continue;

    // 2. Ability Filtering (Catching Evasion Clause)
    const legalAbilities = {};
    for (const [key, abilityName] of Object.entries(species.abilities)) {
        const problems = validator.validateSet({
            species: species.name, name: species.name, ability: abilityName,
            moves: ['Pound'], level: 100
        });
        if (!problems || !problems.some(p => p.includes(abilityName))) {
            legalAbilities[key] = abilityName;
        }
    }
    if (Object.keys(legalAbilities).length === 0) continue;

    // 3. Move Filtering (The Compatibility/Banned Move fix)
    const legalMoves = [];
    const firstAbility = Object.values(legalAbilities)[0];

    // getLearnset helper (simulated)
    let moveIds = [];
    let curr = species;
    while (curr && curr.exists) {
        const ls = dex.data.Learnsets[curr.id]?.learnset;
        if (ls) moveIds.push(...Object.keys(ls));
        curr = dex.species.get(curr.prevo);
    }
    moveIds = [...new Set(moveIds)];

    for (const moveId of moveIds) {
        const move = dex.moves.get(moveId);

        // Strict Compatibility: Ignore 'Past' and 'LGPE' moves to avoid illegal combos
        if (move.isNonstandard === 'Past' || move.isNonstandard === 'LGPE') continue;

        const moveProblems = validator.validateSet({
            species: species.name, name: species.name,
            ability: firstAbility, moves: [move.name], level: 100
        });

        if (!moveProblems || !moveProblems.some(p => p.includes(move.name))) {
            legalMoves.push(move.name);
        }
    }

    if (legalMoves.length >= 4) {
        output[species.name] = {
            abilities: legalAbilities,
            moves: legalMoves.sort(),
            tier: tier
        };
    }
}

fs.writeFileSync(`${formatId}.json`, JSON.stringify(output, null, 4), 'utf8');
console.log(`✅ Success! Data exported to ${formatId}.json`);