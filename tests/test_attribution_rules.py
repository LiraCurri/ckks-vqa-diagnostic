"""Logic of the frozen attribution rules, on synthetic records only."""

import attribution_rules as r
from attribution_rules import Contrast, Evidence


def test_no_shortfall_is_none() -> None:
    assert r.attribute(Evidence(observed=1.05, class_optimum=1.0)) == "none"


def test_prior_accounts_for_the_gap() -> None:
    e = Evidence(observed=100.0, reference="full", class_optimum=99.0, unrestricted_optimum=1.0)
    assert r.attribute(e) == "prior"


def test_coarse_threshold_closed_by_normalising_is_optimiser() -> None:
    e = Evidence(observed=20.0, class_optimum=1.0, scale_check_fails=True, normalised_result=1.2,
                 inverse_best=1.0)
    assert r.attribute(e) == "optimiser"


def test_unreachable_and_not_closed_is_reachability() -> None:
    e = Evidence(observed=20.0, class_optimum=1.0, scale_check_fails=True, normalised_result=19.0,
                 inverse_best=15.0)
    assert r.attribute(e) == "reachability"


def test_inverse_best_within_tolerance_is_optimiser() -> None:
    e = Evidence(observed=20.0, class_optimum=1.0, inverse_best=1.08)
    assert r.attribute(e) == "optimiser"


def test_inverse_best_outside_tolerance_is_reachability() -> None:
    e = Evidence(observed=20.0, class_optimum=1.0, inverse_best=1.5)
    assert r.attribute(e) == "reachability"


def test_readout_explained_by_variance() -> None:
    e = Evidence(observed=5.0, class_optimum=1.0, exact_result=1.02, noise_ratio=0.99)
    assert r.attribute(e) == "readout"


def test_without_components_4_and_5_the_floor_is_misread() -> None:
    e = Evidence(observed=20.0, class_optimum=1.0, scale_check_fails=True, normalised_result=1.2,
                 inverse_best=1.0, optimizer_success=True, restarts_agree=True)
    e = r.withhold(r.withhold(e, 4), 5)
    assert r.attribute(e) == "reachability"     # the conventional, wrong reading


def test_without_components_4_and_5_disagreeing_restarts_are_undetermined() -> None:
    e = Evidence(observed=20.0, class_optimum=1.0, scale_check_fails=True, normalised_result=1.2,
                 inverse_best=1.0, optimizer_success=True, restarts_agree=False)
    e = r.withhold(r.withhold(e, 4), 5)
    assert r.attribute(e) == "undetermined"     # the triangle case: restarts span 318-fold


def test_flags_not_computed_never_give_the_conventional_reading() -> None:
    e = Evidence(observed=20.0, class_optimum=1.0)
    assert r.attribute(e) == "undetermined"


def test_either_component_alone_identifies_the_optimiser() -> None:
    e = Evidence(observed=20.0, class_optimum=1.0, scale_check_fails=True, normalised_result=1.2,
                 inverse_best=1.0)
    assert r.attribute(r.withhold(e, 4)) == "optimiser"
    assert r.attribute(r.withhold(e, 5)) == "optimiser"


def test_without_component_1_the_prior_is_not_separated() -> None:
    e = Evidence(observed=100.0, reference="full", class_optimum=99.0, unrestricted_optimum=1.0,
                 null_result=99.5)
    assert r.attribute(r.withhold(e, 1)) == "none"   # measured against the null, no gap is seen


def test_component_2_changes_nothing_when_component_1_is_present() -> None:
    e = Evidence(observed=20.0, class_optimum=1.0, null_result=1.0, inverse_best=1.0)
    assert r.attribute(r.withhold(e, 2)) == r.attribute(e)


def test_contrast_rules() -> None:
    assert r.attribute_contrast(Contrast(resolved=False, structural_prediction="differ")) == "none"
    assert r.attribute_contrast(Contrast(resolved=True, structural_prediction="differ")) == "structure"
    assert r.attribute_contrast(Contrast(resolved=True, structural_prediction="equivalent")) == "defect"
    assert r.attribute_contrast(Contrast(resolved=True, structural_prediction="none")) == "architecture"
    withheld = r.withhold(Contrast(resolved=True, structural_prediction="equivalent"), 3)
    assert r.attribute_contrast(withheld) == "architecture"


def test_agreement_and_success_rules() -> None:
    assert r.restarts_agree_rule([1.0, 1.1, 1.2, 1.3]) is True
    assert r.restarts_agree_rule([1.0, 1.0, 50.0, 60.0]) is False
    # the boundary: interquartile range just inside and just outside half the median
    assert r.restarts_agree_rule([0.75, 0.75, 1.24, 1.24]) is True
    assert r.restarts_agree_rule([0.75, 0.75, 1.26, 1.26]) is False
    # a 318-fold spread does not agree
    assert r.restarts_agree_rule([2.2e-6, 1.0e-5, 4.8e-5, 9.0e-5, 7.0e-4]) is False
    assert r.optimizer_success_rule([True, True, False]) is True
    assert r.optimizer_success_rule([True, False, False]) is False


def test_degenerate_record_is_undetermined() -> None:
    assert r.attribute(Evidence(observed=0.0, class_optimum=1.0)) == "undetermined"
    assert r.attribute(Evidence(observed=2.0, class_optimum=0.0)) == "undetermined"
    assert r.attribute(Evidence(observed=2.0, class_optimum=0)) == "undetermined"   # integer zero


def test_missing_every_reference_is_undetermined() -> None:
    assert r.attribute(Evidence(observed=5.0)) == "undetermined"
