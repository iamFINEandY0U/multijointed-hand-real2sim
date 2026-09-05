# Vendored dependencies

## cma/  (pycma 4.4.4)

- Source: https://github.com/CMA-ES/pycma  (`pip download cma==4.4.4`, sdist `src/cma`)
- License: BSD 3-Clause (see `cma/LICENSE`)
- Only runtime dependency: `numpy` (present in the Isaac Sim python env).
- Used by `allegro_twin_tuner/cma_tuning.py` via `cma.CMAEvolutionStrategy` (ask/tell).
- Unmodified. To upgrade: replace the `cma/` folder with a newer sdist's `src/cma`
  and re-run `scratchpad/test_cma_tuning.py`.

`cma_tuning._import_cma()` prepends this `_vendor/` dir to `sys.path` so
`import cma` resolves here without a pip install.
