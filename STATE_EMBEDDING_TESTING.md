# Ways to test state embedding

Use these checks when you modify `embedding.py` or `env_wrapper.py`.

## 1) Regression/unit checks

Run:

```bash
python -m unittest tests.test_state_embedding
```

What this validates:
- `embed_move` returns the expected fixed length (`MOVE_EMBED_LEN`) and bounded values.
- Edge move inputs with missing/nullable fields still embed into valid finite vectors.
- Type immunity encoding still maps to `-1.0` in `calc_types_vector`.
- `PokemonRLWrapper.embed_battle` output length is derived from code structure/constants (not a hardcoded `51`).
- When moves are unavailable, unused move slots are zero-padded.
- `print_state` total dimensions are verified dynamically.

## 2) Interactive debug output during play

The wrapper already has a readable state dump:

```python
self.env.print_state(battle)
```

Use it to spot-check:
- Incorrect scaling (values outside expected range)
- Wrong block ordering after adding/removing features
- Missing move embedding slots when no moves are available

## 3) Invariant checks to add when extending features

When you add fields to the embedding:
- Update the observation space shape in `env_wrapper.py`.
- Add a corresponding test assertion for vector length.
- Prefer deriving expected length from constants/feature-block sizes instead of hardcoded totals.
- If the new feature has a known range, assert that range in unit tests.

## 4) Suggested smoke check in training loop

After `env.reset()`, grab one observation and verify:
- Shape is exactly what the model expects
- All values are finite (`np.isfinite(obs).all()`)
- No unexpected all-zero blocks unless intentionally designed

This catches wiring mistakes before a long training run.
