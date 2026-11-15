import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mlpf.conf import SetDecoderConfig
from mlpf.model.set_decoder import ParticleSetDecoder
from test_set_matching import loss, make_raw, make_targets

DIM, NUM_CLASSES = 16, 6


def make_inputs(batch_size=2, num_elements=12, seed=0, num_tracks=4):
    """num_tracks tracks (type 1) then calorimeter elements (type 2), with energies."""
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(batch_size, num_elements, 17, generator=g)
    X[..., 0] = 2.0
    X[:, :num_tracks, 0] = 1.0
    X[..., 1] = X[..., 1].abs() + 0.1
    X[..., 5] = X[..., 5].abs() + 0.1
    phi = torch.rand(batch_size, num_elements, generator=g) * 6.2 - 3.1
    X[..., 3], X[..., 4] = torch.sin(phi), torch.cos(phi)
    memory = torch.randn(batch_size, num_elements, DIM, generator=g)
    return X, X[..., 0] != 0, memory


def calo_inputs(eta, phi, energy, num_tracks=0, seed=0):
    """One event: num_tracks tracks, then calorimeter elements at (eta, phi) with energies in column 5."""
    g = torch.Generator().manual_seed(seed)
    n = num_tracks + len(eta)
    X = torch.zeros(1, n, 17)
    X[0, :, 0] = 2.0
    X[0, :num_tracks, 0] = 1.0
    X[0, :num_tracks, 1] = 1.0
    X[0, :num_tracks, 4] = 1.0
    X[0, num_tracks:, 2] = torch.as_tensor(eta, dtype=torch.float32)
    X[0, num_tracks:, 3] = torch.sin(torch.as_tensor(phi, dtype=torch.float32))
    X[0, num_tracks:, 4] = torch.cos(torch.as_tensor(phi, dtype=torch.float32))
    X[0, num_tracks:, 5] = torch.as_tensor(energy, dtype=torch.float32)
    X[0, num_tracks:, 1] = X[0, num_tracks:, 5] / torch.cosh(X[0, num_tracks:, 2]) * torch.rand(len(eta), generator=g) * 10
    return X, X[..., 0] != 0, torch.randn(1, n, DIM, generator=g)


def build_decoder(seed=0, **overrides):
    torch.manual_seed(seed)
    config = SetDecoderConfig(num_slots=overrides.pop("num_slots", 6), num_layers=2, num_heads=2, query_init="input-conditioned", **overrides)
    return ParticleSetDecoder(DIM, NUM_CLASSES, config)


def brute_anchors(X, step=0.05, num_phi=128, eta_max=4.5, radius=0.1, count=None):
    """Every grid bin, its L-inf energy score, ranked by score then bin order; as numpy (eta, phi, score)."""
    half = math.ceil(eta_max / step)
    ce = np.repeat((np.arange(2 * half) - half + 0.5) * step, num_phi)
    cp = np.tile(-math.pi + (np.arange(num_phi) + 0.5) * 2 * math.pi / num_phi, 2 * half)
    calo = X[0][X[0, :, 0] == 2].double().numpy()
    eta, phi = calo[:, 2], np.arctan2(calo[:, 3], calo[:, 4])
    dphi = np.abs(np.remainder(phi[None] - cp[:, None] + math.pi, 2 * math.pi) - math.pi)
    score = ((np.abs(eta[None] - ce[:, None]) <= radius) & (dphi <= radius)) @ calo[:, 5]
    order = sorted(range(len(ce)), key=lambda i: (-round(score[i], 4), i))[:count]
    return ce[order], cp[order], score[order]


def test_seed_index_names_each_slots_seed_element():
    decoder = build_decoder(num_slots=6)
    X, mask, memory = make_inputs(batch_size=2, num_elements=5, seed=3)
    mask[1, 3:] = False
    seed_index = decoder(memory, mask, X).seed_index
    assert (seed_index[0, :5] >= 0).all() and (seed_index[0, 5:] == -1).all()
    assert (seed_index[1, :3] >= 0).all() and (seed_index[1, 3:] == -1).all()
    learned = ParticleSetDecoder(DIM, NUM_CLASSES, SetDecoderConfig(num_slots=6, num_layers=2, num_heads=2))
    assert learned(memory, mask, X).seed_index is None


def test_strict_seeding_gives_every_track_a_slot_and_raises_over_capacity():
    X, mask, memory = make_inputs(batch_size=1, num_elements=6, seed=6, num_tracks=2)
    X[0, :2, 2] = torch.tensor([0.51, 0.52])
    X[0, :2, 3], X[0, :2, 4] = 0.0, 1.0
    default = build_decoder(num_slots=4, proposal_mode="grid-diverse")(memory, mask, X).seed_index[0, :2]
    strict = build_decoder(num_slots=4, proposal_mode="grid-diverse", strict_seeding=True)(memory, mask, X).seed_index[0, :2]
    assert sorted(strict.tolist()) == [0, 1] and sorted(default.tolist()) != [0, 1]
    X3, mask3, memory3 = make_inputs(batch_size=1, num_elements=6, seed=6, num_tracks=3)
    with pytest.raises(ValueError, match="3 tracks but only 2 tracker slots"):
        build_decoder(num_slots=4, strict_seeding=True)(memory3, mask3, X3)
    with pytest.raises(ValueError, match="strict_seeding and anchor_mode require"):
        SetDecoderConfig(strict_seeding=True)


def test_strict_seeding_raises_when_targets_exceed_seeded_slots():
    targets = make_targets(batch_size=2, num_targets=3, seed=7)
    targets[1, 2:] = 0
    raw = make_raw(2, 5)
    seed_index = torch.tensor([[0, 1, 2, -1, -1], [0, 1, -1, -1, -1]])
    _, losses, _ = loss(targets, raw, seed_index=seed_index, strict_seeding=True)
    torch.testing.assert_close(losses["SeededSlots"], torch.tensor(2.5))
    torch.testing.assert_close(losses["UnseededSlots"], torch.tensor(2.5))
    batch = SimpleNamespace(target_mask=targets[..., 0] != 0, event_id=torch.tensor([640, 641]))
    with pytest.raises(ValueError, match=r"Batch entry 1 \(event_id 641\) has 2 targets but only 1 seeded slots of 5"):
        loss(targets, raw, batch, seed_index=torch.tensor([[0, 1, 2, -1, -1], [0, -1, -1, -1, -1]]), strict_seeding=True)
    # anchors count as seeded
    is_anchor = torch.tensor([[False] * 5, [False, True, False, False, False]])
    loss(targets, raw, seed_index=torch.tensor([[0, 1, 2, -1, -1], [0, -1, -1, -1, -1]]), is_anchor=is_anchor, strict_seeding=True)


def test_anchor_score_is_the_smoothed_calorimeter_energy():
    decoder = build_decoder(anchor_mode="fill_unseeded")
    g = torch.Generator().manual_seed(13)
    eta, phi, energy = torch.rand(60, generator=g) * 2 - 1, torch.rand(60, generator=g) * 6.2 - 3.1, torch.rand(60, generator=g) * 3
    X, mask, _ = calo_inputs(eta, phi, energy, num_tracks=5)
    X[0, :5, 5] = 100.0  # track energy never counts
    a_eta, a_phi, score = decoder._energy_anchors(X[0], mask[0] & (X[0, :, 0] == 2))
    assert len(a_eta) == 180 * 128
    ce, cp, ref = brute_anchors(X)
    got = np.lexsort((a_phi.numpy(), a_eta.numpy()))
    want = np.lexsort((cp, ce))
    assert np.allclose(score.numpy()[got], ref[want], atol=1e-4) and score.max() > 0


def test_fill_unseeded_takes_the_free_slots_in_rank_order():
    decoder = build_decoder(num_slots=40, tracker_query_fraction=0.25, anchor_mode="fill_unseeded")
    g = torch.Generator().manual_seed(14)
    eta, phi, energy = torch.rand(12, generator=g) * 2 - 1, torch.rand(12, generator=g) * 2 - 1.7, torch.rand(12, generator=g) * 4
    X, mask, memory = calo_inputs(eta, phi, energy, num_tracks=3)
    slots, references, _, _, _, (seed_index, is_anchor) = decoder._input_conditioned_queries(memory, mask, X, return_seeding=True)
    assert (seed_index[0, :15] >= 0).all() and (seed_index[0, 15:] == -1).all()
    assert is_anchor[0].tolist() == [False] * 15 + [True] * 25
    ce, cp, _ = brute_anchors(X, count=25)
    assert np.allclose(references[0, 15:, 0].numpy(), ce, atol=1e-5) and np.allclose(references[0, 15:, 1].numpy(), cp, atol=1e-5)
    position = torch.stack([references[0, 15, 0], torch.sin(references[0, 15, 1]), torch.cos(references[0, 15, 1])])
    torch.testing.assert_close(slots[0, 15], decoder.queries[0, 15] + decoder.anchor_embedding + decoder.reference_embedding(position))


def test_anchor_embedding_gets_a_gradient_without_free_slots():
    decoder = build_decoder(num_slots=4, anchor_mode="fill_unseeded")
    X, mask, memory = make_inputs(batch_size=1, num_elements=8, seed=2)
    out = decoder(memory, mask, X)
    assert not out.is_anchor.any()
    sum(o.float().sum() for o in out).backward()
    assert decoder.anchor_embedding.grad is not None


def test_no_anchor_parameters_or_slots_by_default():
    decoder = build_decoder()
    assert decoder.anchor_embedding is None and "anchor_embedding" not in decoder.state_dict()
    X, mask, memory = make_inputs()
    assert not decoder(memory, mask, X).is_anchor.any()


def test_a_neighbor_of_a_brighter_hit_is_seeded_only_after_the_maxima():
    # one track, then calorimeter hits 1 (E 5) and 2 (E 4) 0.01 apart in the ECAL and an isolated hit 3 (E 1)
    X = torch.zeros(1, 4, 17)
    X[0, :, 0] = torch.tensor([1.0, 2, 2, 2])
    X[0, :, 5] = torch.tensor([10.0, 5, 4, 1])
    X[0, :, 1] = X[0, :, 5]
    eta, phi = torch.tensor([0.0, 1.0, 1.01, -1.0]), torch.tensor([0.0, 0.5, 0.5, 2.0])
    X[0, :, 2], X[0, :, 3], X[0, :, 4] = eta, torch.sin(phi), torch.cos(phi)
    X[0, 1:, 10] = X[0, 1:, 5]
    mask = torch.ones(1, 4, dtype=torch.bool)

    def seeded(radius):
        out = build_decoder(num_slots=3, calo_seed_local_max_radius=radius)(torch.randn(1, 4, DIM), mask, X)
        return set(out.seed_index[0][out.seed_index[0] >= 0].tolist())

    # 3 slots: 2 tracker slots, 1 used, so the calorimeter gets 2
    assert seeded(None) == {0, 1, 2}
    assert seeded(0.02) == {0, 1, 3}
    assert seeded(0.005) == {0, 1, 2}


def test_local_max_seeds_fill_every_slot():
    X, mask, memory = make_inputs(batch_size=2, num_elements=40, seed=3)
    X[..., 2] = X[..., 2] * 0.05
    X[..., 10] = torch.where(torch.arange(40) % 2 == 0, X[..., 5], 0.0)
    X[..., 11] = torch.where(torch.arange(40) % 2 == 1, X[..., 5], 0.0)
    out = build_decoder(num_slots=24, calo_seed_local_max_radius=0.05)(memory, mask, X)
    for b in range(2):
        seeds = out.seed_index[b]
        assert (seeds >= 0).all() and len(set(seeds.tolist())) == 24
    with pytest.raises(ValueError, match="calo_seed_local_max_radius"):
        SetDecoderConfig(calo_seed_local_max_radius=0.02)
