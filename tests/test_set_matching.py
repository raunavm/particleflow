from types import SimpleNamespace

import pytest
import torch

from mlpf.model.set_losses import hungarian_match, set_mlpf_loss
from mlpf.model.utils import unpack_predictions, unpack_target

REGRESSION_WEIGHTS = {feature: 1.0 for feature in ("pt", "eta", "sin_phi", "cos_phi", "energy")}
NUM_CLASSES = 6


def make_targets(batch_size=1, num_targets=2, seed=0, ispu=None):
    g = torch.Generator().manual_seed(seed)
    y = torch.zeros(batch_size, num_targets, 14)
    phi = torch.rand(batch_size, num_targets, generator=g) * 6.2 - 3.1
    y[..., 0] = torch.randint(1, 6, (batch_size, num_targets), generator=g).float()
    y[..., 2] = torch.randn(batch_size, num_targets, generator=g)
    y[..., 3] = torch.rand(batch_size, num_targets, generator=g) * 4 - 2
    y[..., 4], y[..., 5] = torch.sin(phi), torch.cos(phi)
    y[..., 6] = y[..., 2] + 0.5
    y[..., 7] = 0.0 if ispu is None else ispu
    y[..., 13] = torch.arange(1, num_targets + 1).float()
    return y


def make_raw(batch_size, num_slots, seed=1, requires_grad=True):
    g = torch.Generator().manual_seed(seed)
    raw = [torch.randn(batch_size, num_slots, d, generator=g) for d in (2, NUM_CLASSES, 5, 2)]
    return [r.requires_grad_(requires_grad) for r in raw]


def loss(targets_tensor, raw, batch=None, **kwargs):
    batch = batch or SimpleNamespace(target_mask=targets_tensor[..., 0] != 0)
    return set_mlpf_loss(unpack_target(targets_tensor, None), unpack_predictions(raw), batch, REGRESSION_WEIGHTS, **kwargs)


def test_unmatched_targets_flag_changes_nothing_when_targets_fit():
    targets = make_targets(batch_size=2, num_targets=4, seed=1)
    results = []
    for flag in (False, True):
        raw = make_raw(2, 6)
        total, losses, _ = loss(targets, raw, allow_unmatched_targets=flag)
        total.backward()
        results.append((total, losses, [r.grad for r in raw[:3]]))
    (t0, l0, g0), (t1, l1, g1) = results
    assert torch.equal(t0, t1) and all(torch.equal(l0[k], l1[k]) for k in l0)
    assert all(torch.equal(a, b) for a, b in zip(g0, g1))
    assert l1["UnmatchedTargets"] == 0 and "UnmatchedTargets" not in l0


def test_extra_targets_go_unmatched_and_get_no_loss():
    targets = make_targets(batch_size=2, num_targets=6, seed=2)
    targets[1, 3:] = 0
    raw = make_raw(2, 4)
    with pytest.raises(ValueError, match="6 targets but the decoder has only 4 slots"):
        loss(targets, raw)
    _, losses, _ = loss(targets, raw, allow_unmatched_targets=True)
    assert losses["UnmatchedTargets"] == 2
    torch.testing.assert_close(losses["UnmatchedTargetFraction"], torch.tensor(2 / 9))
    matches = hungarian_match(unpack_target(targets, None), unpack_predictions(raw), targets[..., 0] != 0, allow_unmatched_targets=True)
    assert [len(s) for s, _ in matches] == [4, 3]
    kept = targets.clone()
    unmatched = torch.ones(6, dtype=torch.bool)
    unmatched[matches[0][1]] = False
    kept[0, unmatched] = 0
    _, reference, _ = loss(kept, raw)
    for key in ("Classification", "Classification_binary", "Regression_pt", "Regression_energy"):
        torch.testing.assert_close(losses[key], reference[key], rtol=0, atol=0)


@pytest.mark.parametrize("allow", [False, True])
def test_threaded_matching_equals_serial(allow):
    targets = make_targets(batch_size=8, num_targets=12 if allow else 9, seed=11)
    targets[3, 5:] = 0
    targets[6] = 0
    raw = make_raw(8, 10, requires_grad=False)
    args = (unpack_target(targets, None), unpack_predictions(raw), targets[..., 0] != 0)
    serial = hungarian_match(*args, allow_unmatched_targets=allow)
    for threads in (2, 8):
        threaded = hungarian_match(*args, allow_unmatched_targets=allow, num_threads=threads)
        assert all(torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) for a, b in zip(serial, threaded))


def test_presence_normalization_by_slot_count():
    targets = make_targets(batch_size=2, num_targets=3, seed=3)
    raw = make_raw(2, 5)
    _, mean, _ = loss(targets, raw)
    _, per_slot, _ = loss(targets, raw, presence_normalization="num_slots")
    torch.testing.assert_close(mean["Classification_binary"], per_slot["Classification_binary"])
    _, weighted, _ = loss(targets, raw, no_object_weight=0.5, presence_normalization="num_slots")
    logits = unpack_predictions(raw)["cls_binary"]
    presence = torch.zeros(2, 5, dtype=torch.long)
    for event, (slots, _) in enumerate(hungarian_match(unpack_target(targets, None), unpack_predictions(raw), targets[..., 0] != 0)):
        presence[event, slots] = 1
    ce = torch.nn.functional.cross_entropy(logits.reshape(-1, 2), presence.reshape(-1), weight=torch.tensor([0.5, 1.0]), reduction="sum") / 10
    torch.testing.assert_close(weighted["Classification_binary"], ce)
