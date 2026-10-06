"""The exact gradient of the gradient cross-check, and the topology study's
ring factorisation and reachable rank."""

import numpy as np

import gradient_cross_check as gcc
import topology_rank_check as trc


def test_parameter_shift_gradient_matches_central_differences() -> None:
    for n_layers in (1, 3):
        assert gcc.validate_gradient(gcc.Cell(n_layers), n_points=2) < 1e-9


def test_even_family_scales_have_zero_gradient_for_the_equivariant_model() -> None:
    cell = gcc.Cell(1)
    p = gcc.sge.random_params(1, 5)
    _, g = cell.objective_and_gradient(p)
    # the two even-family log-scales cannot move an equivariant model's output
    assert np.all(g[cell.nc + 2:] == 0.0)


def test_ring_factorisation_follows_proposition_3() -> None:
    fac = trc.factorisation_check((1, 2), n_points=2, seed=3)
    assert (fac.ring_correlators_factorised == fac.predicted_by_proposition_3).all()
    assert set(fac[fac.topology == "crossed"].ring_correlators_factorised) == {4}
    assert set(fac[fac.topology != "crossed"].ring_correlators_factorised) == {2}


def test_crossed_circuit_reaches_lower_rank_than_cycle_circuits() -> None:
    ranks = trc.rank_study((2,), n_points=2, eps=1e-6, rank_rtol=1e-8, seed=3)
    cycle = ranks[ranks.topology != "crossed"].rank_max
    crossed = ranks[ranks.topology == "crossed"].rank_max
    assert cycle.min() > crossed.max()


def test_structure_matched_null_is_a_normalised_pair_product() -> None:
    import topology_structured_null as tsn
    rng = np.random.default_rng(4)
    x = rng.normal(size=tsn.N_STATE + 4)
    for topology, pairs in tsn.topo.PAIR_TOPOLOGIES.items():
        state = tsn.product_state(x, topology)
        assert np.isclose(np.vdot(state, state).real, 1.0)
        ev = lambda o: float(np.real(state.conj() @ tsn.topo.pauli_operator(o) @ state))  # noqa: E731
        # a correlator joining the two pairs factorises
        a, c = pairs[0][0], pairs[1][0]
        za, zc = tsn.topo.pauli_string({a: "Z"}), tsn.topo.pauli_string({c: "Z"})
        assert abs(ev(tsn.topo.pauli_string({a: "Z", c: "Z"})) - ev(za) * ev(zc)) < 1e-12


def test_structure_matched_null_has_the_circuit_rank() -> None:
    import topology_structured_null as tsn
    ranks = trc.rank_study((2,), n_points=2, eps=1e-6, rank_rtol=1e-8, seed=3)
    for topology, alignment in (("nearest", "nearest"), ("crossed", "crossed"), ("crossed", "nearest")):
        circuit = int(ranks[(ranks.topology == topology) & (ranks.decoder_alignment == alignment)].rank_max.max())
        assert tsn.null_rank(topology, alignment, n_points=2) == circuit


def test_structure_matched_null_gradient_is_exact() -> None:
    import topology_structured_null as tsn
    assert tsn.validate_gradient(n_points=1) < 1e-6


def test_null_cross_check_gradient_is_exact():
    """The analytic gradient of the spherical null agrees with central differences."""
    import null_cross_check as ncc
    assert ncc.validate_gradient(ncc.Null()) < 1e-7
