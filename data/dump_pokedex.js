const { Dex } = require('../pokemon-showdown/dist/sim/dex');
const fs = require('fs');

const formatId = 'gen9nationaldex';
const dex = Dex.forFormat(formatId);
const output = {};

console.log(`Targeting Format: ${formatId}`);

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
                    moves.add(dex.moves.get(moveId).name);
                }
            }
            currentSpecies = dex.species.get(currentSpecies.prevo);
        }

        output[species.name] = {
            // This now preserves the {"0": "...", "H": "..."} structure
            abilities: species.abilities,
            moves: Array.from(moves).sort()
        };
    }
}

fs.writeFileSync(`${formatId}.json`, JSON.stringify(output, null, 4));
console.log(`✅ Success! Generated ${formatId}.json with ${Object.keys(output).length} Pokémon.`);