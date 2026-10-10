# A Diagnostic Protocol for Evaluating Classical-Output Variational Quantum Algorithms

Code, data and analysis for the paper *A Diagnostic Protocol for Evaluating
Classical-Output Variational Quantum Algorithms: A CKKS Bootstrapping Case
Study* (E. Shaska and D.-K. Kim, Oakland University).

Every number in the paper is recomputed from the stored results by
`notebooks/paper2_analysis.ipynb`, which writes them to `paper/paper_values.tex`
as `\pv...` LaTeX macros, together with two generated tables
(`paper/family_table.tex`, `paper/topology_table.tex`). All quantum circuits are
simulated exactly with NumPy; no quantum SDK or hardware is required.

## Layout

```
src/
  Principal experiment (sine basis, half-period grading)
    sector_graded_evalmod.py            ansatz families, decoder, closed-form references,
                                        dimension-matched nulls, executable checks,
                                        Jacobian ranks, finite-shot readout
    precision_plateau_diagnostic.py     direct inverse problem, conditioning, tolerance sweep
    optimizer_cross_check.py            restart diagnostics (same and lowered tolerance)
    objective_scale_check.py            stopping test normalised by the objective's scale
    gradient_cross_check.py             exact (parameter-shift) gradient vs finite differences
    null_cross_check.py                 the same for the classical null
    jacobian_decoder_ablation.py        Jacobian rank by parameter block (circuit vs decoder)
    noisy_target_experiment.py          noisy training targets, six arms

  Topology study (Chebyshev basis)
    evalmod_factorial_observable_study.py   3 x 2 x 3 factorial; sigma-built mirror cells;
                                            relabelling-equivalence check
    topology_aware_evalmod_experiments_v3.py shared datasets, baselines and gates
    topology_rank_check.py              ring-correlator factorisation and Jacobian ranks
    topology_structured_null.py         structure-matched null (two two-qubit states)

  Calibration of the protocol (RQ2)
    attribution_rules.py                attribution rules v1.3, frozen at commit ceafb32
    ablation_records.py                 evidence records from stored results; executed ablation
    contrast_records.py                 contrast rule applied to the topology study
    constructed_cases.py                constructed optimiser and reachability cases
    representation_limit_case.py        certified representation limit (frozen decoder scales)
    threshold_sensitivity.py            rules re-applied over a prespecified threshold grid
    prospective_rule_change.py          candidate rule v1.4, scored but not applied
    mutation_matrix.py                  mutation testing of the executable checks

tests/                                  structural checks, fault-injection tests, relabelling tests
notebooks/paper2_analysis.ipynb         analysis; writes paper/paper_values.tex and the tables
results/                                stored outputs of every run below
paper/                                  generated macros and tables (the manuscript source is not
                                        included)
```

## Setup

```
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The reported results were produced with Python 3.11.9, numpy 2.4.1,
scipy 1.17.1 and pandas 3.0.0 (`results/environment_analysis.json`).
Evaluation counts and per-seed endpoints depend on the L-BFGS-B implementation,
so other SciPy versions reproduce the medians but not every restart.

## Tests

```
pytest            # structural checks, fault injection, relabelling tests (~3 s)
pytest -m slow    # also the restart test for the stopping-rule defect (~40 s)
```

- `tests/test_sector_validators.py` checks the decoder table, selection rule,
  basis rotations, Jacobian rank bound and reference optima.
- `tests/test_fault_injection.py` reproduces each defect found during the study
  (D1–D4) and checks that the corresponding check detects it and passes on the
  correct code.
- `tests/test_factorial_relabelling.py` checks Proposition 1 as implemented:
  every nearest cell and its sigma-image in the mirror block are the same
  optimisation problem, and the two ways the original code broke this are
  detected.
- `tests/test_attribution_rules.py` and `tests/test_gradient_and_topology.py`
  cover the attribution rules and the gradient and topology checks.

## Reproducing the results

Run from the repository root, in this order; later steps read the outputs of
earlier ones and check at start-up that they reproduce them. Times are
approximate on one CPU core.

### Principal experiment

| Step | Command | Output | Time |
|---|---|---|---|
| Sector study, both decoders | `python src/sector_graded_evalmod.py --full --output-root results/sector_results_full` | `results/sector_results_full/` | hours |
| Precision-plateau diagnostic | set `RUN_DIAG = True` in the notebook | `results/precision_plateau_diagnostic/` | ~1.5 h |
| Restart, same tolerance | `python src/optimizer_cross_check.py` | `results/optimizer_cross_check/idempotence_test.csv` | minutes |
| Restart, tolerance lowered | `python src/optimizer_cross_check.py --restart-ftol 1e-18` | `.../idempotence_test_ftol1e-18.csv` | minutes |
| Normalised stopping test | `python src/objective_scale_check.py` | `results/objective_scale/` | ~1 h |
| Same, generic family | `python src/objective_scale_check.py --family generic --target sawtooth` (and `double_sawtooth`) | `results/objective_scale/` | ~1 h each |
| Gradient cross-check | `python src/gradient_cross_check.py` | `results/gradient_cross_check/gradient_cross_check.csv` | ~1 h |
| Null cross-check | `python src/null_cross_check.py` | `.../null_cross_check.csv` | seconds |
| Jacobian block ablation | `python src/jacobian_decoder_ablation.py` | `results/jacobian_ablation/` | minutes |
| Noisy targets | `python src/noisy_target_experiment.py` | `results/noisy_target/noisy_target_results.csv` | ~20 min |
| Noisy targets, tight settings | `python src/noisy_target_experiment.py --tight --sigmas 1e-3 1e-2 3e-2` | `.../noisy_target_results_tight.csv` | ~1 h |

### Topology study

| Step | Command | Output | Time |
|---|---|---|---|
| Factorial (reported run) | `python src/evalmod_factorial_observable_study.py --full --maxfun 1000000 --output-dir results/factorial_relabelled` | `results/factorial_relabelled/` | hours |
| Structure-matched null | `python src/topology_structured_null.py` | `results/topology_null/` | ~1 h |
| Factorisation and rank check | `python src/topology_rank_check.py` | `results/topology_rank/` | minutes |

`--smoke` runs a small version of the factorial to test the pipeline. The
script checks the relabelling equivalence of Proposition 1 numerically before
any optimisation and stops if it fails. `results/factorial_maxfun/` is the
earlier run with the original mirror cells, kept for the class-level comparison
the notebook prints; its nearest and crossed cells are identical to the
reported run.

### Calibration

| Step | Command | Output | Time |
|---|---|---|---|
| Representation-limit case | `python src/representation_limit_case.py` | `results/representation_limit/` | ~1 h |
| Constructed cases | `python src/constructed_cases.py` | `results/constructed_cases/constructed_cases.json` | ~1 h |
| Evidence records and ablation | `python src/ablation_records.py` | `results/ablation/ablation_table.csv` | seconds |
| Contrast rule on the topology study | `python src/contrast_records.py` | `results/ablation/contrast_table.csv` | seconds |
| Threshold sensitivity | `python src/threshold_sensitivity.py` | `results/threshold_sensitivity/` | seconds |
| Prospective rule v1.4 (scored, not applied) | `python src/prospective_rule_change.py` | `results/prospective_v14/comparison.csv` | seconds |
| Mutation testing of the checks | `python src/mutation_matrix.py` | `results/mutation/mutation_matrix.csv` | minutes |

The attribution rules in `src/attribution_rules.py` (version 1.3) were
committed at `ceafb32` before the constructed cases were generated and have not
been changed since; `git log -- src/attribution_rules.py` shows this.

## Analysis

Open `notebooks/paper2_analysis.ipynb` and run all cells. With the default
configuration it reads the stored results and reruns no experiment; the `RUN_*`
flags in the first code cell rerun individual studies. The notebook also runs
the fast tests, cross-checks its own contrast statistics against the factorial
module, and writes `paper/paper_values.tex`, `paper/family_table.tex` and
`paper/topology_table.tex`.

## What is stored and what is regenerated

All result files the notebook reads are stored in `results/`. The large
per-shot files `factorial_grouped_shot_results.csv` (about 70 MB per run) are
not committed; the factorial script regenerates them, and the notebook uses the
summaries. The manuscript source is not part of this repository.
