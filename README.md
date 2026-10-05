# Diagnostic evaluation of classical-output variational quantum algorithms

Code and analysis for the paper *A Diagnostic Protocol for Evaluating
Classical-Output Variational Quantum Algorithms: A CKKS Bootstrapping Case
Study*. Every number quoted in the paper is recomputed from the
stored results by `notebooks/paper2_analysis.ipynb`, which writes them to
`paper/paper_values.tex` as LaTeX macros.

All quantum circuits are simulated exactly with NumPy; no quantum SDK or
hardware is required.

## Layout

```
src/
  sector_graded_evalmod.py               sector-graded study: ansatz families, decoder,
                                         closed-form references, nulls, invariants,
                                         Jacobian ranks, finite-shot readout
  precision_plateau_diagnostic.py        direct inverse problem, conditioning, ftol sweep
  optimizer_cross_check.py               restart diagnostics (same and lowered tolerance)
  objective_scale_check.py               stopping test normalised by the objective's scale
  jacobian_decoder_ablation.py           Jacobian rank by parameter block
  noisy_target_experiment.py             noisy training targets, four arms
  evalmod_factorial_observable_study.py  topology factorial (Section V)
  topology_aware_evalmod_experiments_v3.py
                                         shared datasets, baselines and gates used by
                                         the factorial study
tests/                                   structural checks and fault-injection tests
notebooks/paper2_analysis.ipynb          analysis; writes paper/paper_values.tex
results/                                 outputs of the runs below
paper/                                   manuscript and generated macros
```

## Setup

```
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Tests

```
pytest            # structural checks and fault-injection tests, about 2 s
pytest -m slow    # also the restart test for the stopping-rule defect, about 70 s
```

`tests/test_fault_injection.py` reproduces each of the four defects described
in the paper (D1-D4) and checks that the corresponding check detects it and
passes on the correct code.

## Reproducing the results

Run from the repository root. Times are approximate, on a single CPU core.

| Step | Command | Output | Time |
|---|---|---|---|
| Sector study, both decoders | `python src/sector_graded_evalmod.py --full --output-root results/sector_results_full` | `results/sector_results_full/` | hours |
| Precision-plateau diagnostic | set `RUN_DIAG = True` in the notebook | `results/precision_plateau_diagnostic/` | ~1.5 h |
| Restart, same tolerance | `python src/optimizer_cross_check.py` | `results/optimizer_cross_check/idempotence_test.csv` | minutes |
| Restart, tolerance lowered | `python src/optimizer_cross_check.py --restart-ftol 1e-18` | `.../idempotence_test_ftol1e-18.csv` | minutes |
| Normalised stopping test | `python src/objective_scale_check.py` | `results/objective_scale/` | ~1 h |
| Jacobian block ablation | `python src/jacobian_decoder_ablation.py` | `results/jacobian_ablation/` | minutes |
| Noisy targets | `python src/noisy_target_experiment.py` | `results/noisy_target/noisy_target_results.csv` | ~20 min |
| Noisy targets, tight settings | `python src/noisy_target_experiment.py --tight --sigmas 1e-3 1e-2 3e-2` | `.../noisy_target_results_tight.csv` | ~1 h |
| Topology factorial | `python src/evalmod_factorial_observable_study.py --full --maxfun 1000000 --output-dir results/factorial_maxfun` | `results/factorial_maxfun/` | hours |

The restart diagnostics, the stopping-test check and the noisy-target experiment read the outputs of
the first two steps, and check at start-up that they reproduce them; run those
first. `evalmod_factorial_observable_study.py --smoke` runs a small version of
the factorial to test the pipeline.

Per-seed endpoints can differ slightly between machines, since runs stop on a
flat region where floating-point details decide the last iterations; the
reported medians reproduce.

## Analysis

Open `notebooks/paper2_analysis.ipynb` and run all cells. With the default
configuration it reads the stored results and reruns no experiment; the
`RUN_*` flags in the first code cell rerun individual studies. The last cell
writes `paper/paper_values.tex` and lists any macro used in `paper/main.tex`
that the notebook does not define.

## Citation

If you use this code, please cite the paper. Repository DOI: to be added.

## License

To be added.
