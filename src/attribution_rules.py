"""Attribution rules for the executed ablation, fixed before any case is generated.

These rules turn the protocol's diagnostic outputs for one case into a single
verdict: the stage to which the shortfall is attributed. They read a record of
numbers and flags; they never run an experiment. The thresholds below are part
of the rules and must not be changed after constructed cases have been seen.
The commit that adds this file is the record of when the rules were fixed.

Evidence fields, by protocol component (None means withheld or unavailable)
--------------------------------------------------------------------------
always available (what any study reports)
    observed            nested RMSE of the reported result (median over restarts)
    reference           "class" or "full": the optimum the question is posed against
    optimizer_success   the optimiser reported convergence on most restarts;
                        computed by optimizer_success_rule
    restarts_agree      restarts give a stable result; computed by
                        restarts_agree_rule
component 1, objective-aligned optima
    class_optimum       nested RMSE of the exact optimum over the model's class
    unrestricted_optimum nested RMSE of the exact optimum over the full space
component 2, matched null
    null_result         nested RMSE of the dimension-matched null, same settings.
                        Used only as a stand-in reference when the exact optima
                        are withheld; where component 1 is present, withholding
                        component 2 changes no verdict, by construction.
component 3, structural accounting
    structural_prediction  for a contrast between two cells: "differ" (the design
                        predicts a structural difference), "equivalent" (it
                        predicts the cells are the same problem), or "none" (no
                        prediction)
component 4, reachability
    inverse_best        nested RMSE of the best point of the direct inverse fit
                        to the base optimum
component 5, optimiser sensitivity
    scale_check_fails   the effective stopping threshold is coarse relative to
                        the data-fit term (check P7)
    normalised_result   nested RMSE after rerunning with the stopping test scaled
                        to the objective
component 6, readout accounting
    exact_result        nested RMSE with exact expectations, same parameters
    noise_ratio         measured coefficient noise over the variance formula

Verdicts: "none", "prior", "readout", "optimiser", "reachability",
"structure", "architecture" (a difference with no structural prediction, or
the conventional reading when structural accounting is withheld), "defect" (a
resolved difference between cells predicted to be equivalent), "undetermined"
(including any record with a non-positive error, which the rules cannot read).
"Reachable" uses the same metric and tolerance as "no shortfall": the inverse
fit's best point is within GAP_TOL of the base optimum in nested RMSE.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields, replace
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

RULES_VERSION = "1.3"

# Thresholds, fixed with the rules.
GAP_TOL = 0.10          # relative nested-RMSE excess treated as no shortfall
SHARE = 0.5             # share of the (log) gap a stage must account for
READOUT_TOL = 0.25      # |noise_ratio - 1| within which readout matches the variance formula
AGREE_FACTOR = 0.5      # restarts agree if their interquartile range is at most this times the
                        # median, i.e. the middle half lies within about +-25% of it
SUCCESS_SHARE = 0.5     # the optimiser "reports success" if more than this share of restarts converged

COMPONENT_FIELDS: Dict[int, Tuple[str, ...]] = {
    1: ("class_optimum", "unrestricted_optimum"),
    2: ("null_result",),
    3: ("structural_prediction",),
    4: ("inverse_best",),
    5: ("scale_check_fails", "normalised_result"),
    6: ("exact_result", "noise_ratio"),
}


@dataclass(frozen=True)
class Evidence:
    observed: float
    reference: str = "class"
    optimizer_success: Optional[bool] = None   # None: not computed
    restarts_agree: Optional[bool] = None      # None: not computed
    class_optimum: Optional[float] = None
    unrestricted_optimum: Optional[float] = None
    null_result: Optional[float] = None
    inverse_best: Optional[float] = None
    scale_check_fails: Optional[bool] = None
    normalised_result: Optional[float] = None
    exact_result: Optional[float] = None
    noise_ratio: Optional[float] = None


@dataclass(frozen=True)
class Contrast:
    """A difference between two design cells."""
    resolved: bool                                 # survives the multiplicity-adjusted test
    structural_prediction: Optional[str] = None    # component 3: "differ", "equivalent", "none"


def restarts_agree_rule(values: Sequence[float]) -> bool:
    """Restarts agree if the interquartile range of their nested RMSE is at most
    AGREE_FACTOR times the median."""
    v = np.asarray(values, dtype=float)
    q25, q50, q75 = np.percentile(v, [25, 50, 75])
    return bool(q75 - q25 <= AGREE_FACTOR * q50)


def optimizer_success_rule(converged: Sequence[bool]) -> bool:
    """The optimiser reports success if more than SUCCESS_SHARE of restarts converged."""
    c = np.asarray(converged, dtype=bool)
    return bool(c.mean() > SUCCESS_SHARE)


def _log_ratio(a: float, b: float) -> float:
    return math.log(a / b)


def _readable(e: Evidence) -> bool:
    """Every numeric error in the record is positive."""
    for f in fields(e):
        v = getattr(e, f.name)
        if (isinstance(v, (int, float)) and not isinstance(v, bool)
                and f.name != "noise_ratio" and not v > 0):
            return False
    return e.observed > 0


def attribute(e: Evidence) -> str:
    """Attribute one case's shortfall to a stage."""
    if not _readable(e):
        return "undetermined"
    # Reference optimum: the exact one if component 1 is available, otherwise
    # the matched null as the best available stand-in.
    if e.reference == "full" and e.unrestricted_optimum is not None:
        ref = e.unrestricted_optimum
    elif e.reference == "class" and e.class_optimum is not None:
        ref = e.class_optimum
    elif e.null_result is not None:
        ref = e.null_result
    else:
        return "undetermined"

    gap = _log_ratio(e.observed, ref)
    if gap <= math.log1p(GAP_TOL):
        return "none"

    # Prior: the class optimum's own gap to the unrestricted optimum.
    base = ref
    if (e.reference == "full" and e.class_optimum is not None
            and e.unrestricted_optimum is not None):
        prior = _log_ratio(e.class_optimum, e.unrestricted_optimum)
        if prior >= SHARE * gap:
            return "prior"
        base = e.class_optimum
    remaining = _log_ratio(e.observed, base)
    if remaining <= math.log1p(GAP_TOL):
        return "prior"

    # Readout: exact expectations reach the base optimum, and the shot result
    # is explained by the variance formula.
    if e.exact_result is not None and e.noise_ratio is not None:
        if (_log_ratio(e.exact_result, base) <= math.log1p(GAP_TOL)
                and abs(e.noise_ratio - 1.0) <= READOUT_TOL):
            return "readout"

    # Optimiser: a coarse stopping threshold, and a normalised rerun closes
    # more than SHARE of the remaining gap.
    closed = None
    if e.normalised_result is not None:
        closed = _log_ratio(e.observed, e.normalised_result) / remaining
    if e.scale_check_fails and closed is not None and closed > SHARE:
        return "optimiser"

    # Reachability: the inverse fit's best point stays outside GAP_TOL of the
    # base optimum, and normalising does not close the gap.
    if e.inverse_best is not None:
        reachable = e.inverse_best <= (1.0 + GAP_TOL) * base
        if not reachable and not (closed is not None and closed > SHARE):
            return "reachability"
        if reachable:
            # reachable, yet not reached: the shortfall lies in optimisation
            return "optimiser"

    # Conventional reading when components 4 and 5 give no verdict: a run that
    # reports convergence and is stable across restarts is read as a model limit.
    # Either flag missing (None) counts as not established.
    if e.optimizer_success and e.restarts_agree:
        return "reachability"
    return "undetermined"


def attribute_contrast(c: Contrast) -> str:
    """Attribute a resolved difference between two design cells."""
    if not c.resolved:
        return "none"
    if c.structural_prediction is None or c.structural_prediction == "none":
        # no structural prediction, or structural accounting withheld
        return "architecture"
    return "structure" if c.structural_prediction == "differ" else "defect"


def withhold(record, component: int):
    """The record with one component's outputs withheld, for the ablation."""
    blank = {f: None for f in COMPONENT_FIELDS[component] if hasattr(record, f)}
    return replace(record, **blank)
