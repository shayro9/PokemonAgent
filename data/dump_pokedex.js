const { Dex } = require('../pokemon-showdown/dist/sim/dex');
const { TeamValidator } = require('../pokemon-showdown/dist/sim/team-validator');
const fs = require('fs');

const formatId = 'gen9nationaldex';
const dex = Dex.forFormat(formatId);
const validator = new TeamValidator(formatId);
const output = {};

console.log(`Targeting Format: ${formatId}`);

// Get the banlist from the format object
const banlist = validator.format.banlist || [];
const restricted = validator.format.restricted || [];

const allSpecies = dex.species.all();

for (const species of allSpecies) {
    if (!species.exists || (species.isNonstandard && !['Past', 'Future'].includes(species.isNonstandard))) {
        continue;
    }

    const speciesData = dex.species.get(species.id);
    const tier = speciesData.tier;

    if (tier !== 'Illegal' && tier !== 'Unreleased' && tier !== 'CAP') {
        const moves = new Set();
        let currentSpecies = species;

        while (currentSpecies && currentSpecies.exists) {
            const learnset = dex.data.Learnsets[currentSpecies.id]?.learnset;
            if (learnset) {
                for (const moveId in learnset) {
                    const move = dex.moves.get(moveId);

                    // FILTER LOGIC:
                    // 1. Is the move in the format's banlist?
                    // 2. Is the move "Nonstandard" (like Z-moves or Max moves in Gen 9)?
                    const isBanned = banlist.includes(move.name) || restricted.includes(move.name);
                    const isIllegal = move.isNonstandard && move.isNonstandard !== 'Past';

                    if (!isBanned && !isIllegal && move.exists) {
                        moves.add(move.name);
                    }
                }
            }
            currentSpecies = dex.species.get(currentSpecies.prevo);
        }

        output[species.name] = {
            abilities: species.abilities,
            moves: Array.from(moves).sort()
        };
    }
}

// Write with UTF-8 encoding explicitly (though Node does this by default)
fs.writeFileSync(`${formatId}.json`, JSON.stringify(output, null, 4), 'utf8');
console.log(`✅ Success! Generated ${formatId}.json without banned moves.`);