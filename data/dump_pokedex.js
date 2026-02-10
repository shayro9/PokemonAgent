const fs = require('fs');

const pokedex = require('../pokemon-showdown/data/pokedex.ts').Pokedex;
const learnsets = require('../pokemon-showdown/data/learnsets.ts').Learnsets;

const result = {};

for (const key in pokedex) {
  if (pokedex.hasOwnProperty(key)) {
    const pokemon = pokedex[key];
    if (pokemon.num <= 0 || pokemon.num > 151) continue;

    const abilities = pokemon.abilities;

    let moves = [];
    if (learnsets[key] && learnsets[key].learnset) {
      moves = Object.keys(learnsets[key].learnset);
    }

    result[key] = {
      abilities: abilities,
      moves: moves
    };
  }
}

fs.writeFileSync('pokemon_data.json', JSON.stringify(result, null, 2));

console.log("Extraction complete! Check pokemon_data.json");