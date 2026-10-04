"""Relabelling equivalence of the topology comparison (Proposition 1).

The mirror circuit is the image of the nearest circuit under sigma and is read
through the sigma-images of the canonical decoders. For every alignment a, the
cells (nearest circuit, decoder a) and (mirror circuit, decoder sigma(a)) must
then be the same optimisation problem up to a permutation of the circuit
parameters. The original code broke this in two ways: the mirror pairs were
coded as (0,3),(1,2), reversing one pair's orientation relative to the image of
the nearest pairs, and the mirror circuit was read through the canonical
decoders, so the same state's expectations fed different coefficient slots.
"""

import pytest

import evalmod_factorial_observable_study as study


def test_relabelling_equivalence_holds_for_every_alignment() -> None:
    study.validate_relabelling_equivalence()


def test_nearest_and_crossed_cells_use_canonical_decoders() -> None:
    for alignment in study.DECODER_ALIGNMENTS:
        for topology in ("nearest", "crossed"):
            assert study.decoder_spec(alignment, topology) is study.DECODER_SPECS[alignment]


def test_one_slot_order_within_each_circuit_block() -> None:
    for specs in (study.DECODER_SPECS, study.DECODER_SPECS_SIGMA):
        fixed = {specs[a].observables[:8] for a in study.DECODER_ALIGNMENTS}
        assert len(fixed) == 1


def test_original_pair_orientation_is_detected(monkeypatch) -> None:
    monkeypatch.setitem(study.PAIR_TOPOLOGIES, "mirror", ((0, 3), (1, 2)))
    with pytest.raises(AssertionError):
        study.validate_relabelling_equivalence(layer_values=(1,), trials=2)


def test_reading_the_mirror_circuit_canonically_is_detected(monkeypatch) -> None:
    monkeypatch.setattr(study, "USE_SIGMA_FRAME", False)
    with pytest.raises(AssertionError):
        study.validate_relabelling_equivalence(layer_values=(1,), trials=2)
