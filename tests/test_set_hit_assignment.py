import math

import pytest
import torch

from mlpf.conf import MLPFConfig, SetDecoderConfig
from mlpf.model.PFDataset import PFBatch
from mlpf.model.mlpf import MLPF
from mlpf.model.set_decoder import ParticleSetDecoder, assigned_energy
from mlpf.model.set_losses import assignment_dice_costs, assignment_dice_loss, hit_grouping_loss, hungarian_match
from mlpf.model.utils import unpack_predictions

DIM, NUM_CLASSES = 16, 6
REGRESSION_WEIGHTS = {feature: 1.0 for feature in ("pt", "eta", "sin_phi", "cos_phi", "energy")}
LATE = dict(query_init="input-conditioned", hit_grouping=True, hit_grouping_late=True, hit_energy_readout=True)


def colliderml_inputs(batch_size=2, num_tracks=3, num_calo=9, seed=0):
    """Tracks (type 1), then calorimeter elements (type 2) with ECAL (col 10) or HCAL (col 11) energy."""
    g = torch.Generator().manual_seed(seed)
    n = num_tracks + num_calo
    X = torch.zeros(batch_size, n, 17)
    X[..., 0] = 2.0
    X[:, :num_tracks, 0] = 1.0
    eta = torch.rand(batch_size, n, generator=g) * 4 - 2
    phi = torch.rand(batch_size, n, generator=g) * 6.2 - 3.1
    energy = torch.rand(batch_size, n, generator=g) * 5 + 0.1
    X[..., 1], X[..., 2], X[..., 3], X[..., 4], X[..., 5] = energy / torch.cosh(eta), eta, torch.sin(phi), torch.cos(phi), energy
    calo = X[..., 0] == 2
    hcal = calo & (torch.arange(n) % 2 == 0)
    X[..., 10] = torch.where(calo & ~hcal, energy, 0.0)
    X[..., 11] = torch.where(hcal, energy, 0.0)
    return X, torch.ones(batch_size, n, dtype=torch.bool)


def grouping_batch():
    """One event: a track (pn 1) and four calorimeter hits, two owned by target pn 1, one by pn 2, one unowned."""
    X = torch.zeros(1, 5, 17)
    X[0, :, 0] = torch.tensor([1.0, 2.0, 2.0, 2.0, 2.0])
    X[0, :, 5] = torch.tensor([3.0, 2.0, 1.0, 4.0, 0.5])
    ytarget = torch.zeros(1, 5, 14)
    ytarget[0, :, 13] = torch.tensor([1.0, 1.0, 1.0, 2.0, 0.0])
    ytarget_set = torch.zeros(1, 2, 14)
    ytarget_set[0, :, 0] = torch.tensor([1.0, 2.0])
    ytarget_set[0, :, 13] = torch.tensor([1.0, 2.0])
    return PFBatch(X=X, ytarget=ytarget, ytarget_set=ytarget_set)


def mlpf_config(dataset="colliderml", **set_decoder):
    d = {
        "dataset": dataset,
        "data_dir": "/tmp",
        "model": {
            "type": "attention",
            "output_mode": "set",
            "input_encoding": "joint",
            "attention": {"num_convs": 1, "num_heads": 2, "head_dim": 8, "use_pre_layernorm": True, "attention_type": "math"},
            "set_decoder": {"num_slots": 8, "num_layers": 2, "num_heads": 2, **set_decoder},
            "hit_feature_engineering": {"enabled": False},
        },
        "conv_type": "attention",
    }
    if dataset == "colliderml":
        d["train_dataset"] = {"colliderml": {"physical": {"samples": {"colliderml_ttbar_pu0_v20_pf": {"version": "1.2.0", "splits": ["1"]}}}}}
        d["set_target_source"] = "gen_target"
    return MLPFConfig.model_validate(d)


def model_batch():
    X, _ = colliderml_inputs(batch_size=1)
    ytarget = torch.zeros(1, X.shape[1], 14)
    ytarget[0, :, 13] = torch.tensor([1, 2, 3, 1, 1, 2, 4, 4, 5, 0, 5, 3], dtype=torch.float32)
    ytarget_set = torch.zeros(1, 5, 14)
    ytarget_set[0, :, 0] = torch.tensor([1.0, 1.0, 1.0, 2.0, 3.0])
    ytarget_set[0, :, 2] = torch.log(torch.tensor([2.0, 3.0, 1.5, 4.0, 2.5]))
    ytarget_set[0, :, 3] = torch.tensor([0.1, -0.5, 1.0, 0.3, -1.2])
    ytarget_set[0, :, 5] = 1.0
    ytarget_set[0, :, 6] = ytarget_set[0, :, 2] + torch.log(torch.cosh(ytarget_set[0, :, 3]))
    ytarget_set[0, :, 13] = torch.arange(1, 6).float()
    return PFBatch(X=X, ytarget=ytarget, ytarget_set=ytarget_set)


def test_config_guards():
    with pytest.raises(ValueError, match="requires hit_grouping_late"):
        SetDecoderConfig(query_init="input-conditioned", hit_grouping=True, hit_energy_readout=True)
    with pytest.raises(ValueError, match="require hit_grouping"):
        SetDecoderConfig(query_init="input-conditioned", hit_grouping_late=True)
    with pytest.raises(ValueError, match="pools nothing"):
        SetDecoderConfig(query_init="input-conditioned", hit_grouping=True, hit_grouping_late=True, tracker_group_readout=True)
    with pytest.raises(ValueError, match="ColliderML element layout"):
        mlpf_config("cld_hits", **LATE)


def test_chunked_assigned_energy_equals_dense_and_is_spent_once():
    X, mask = colliderml_inputs()
    mask[1, -2:] = False
    g = torch.Generator().manual_seed(3)
    keys = torch.nn.functional.normalize(torch.randn(2, X.shape[1], 4, generator=g), dim=-1)
    slots = torch.nn.functional.normalize(torch.randn(2, 5, 4, generator=g), dim=-1)
    background = torch.randn(2, X.shape[1], 1, generator=g)
    out = assigned_energy((keys, slots, background), mask, X, chunk_size=2)
    for b in range(2):
        calo = mask[b] & (X[b, :, 0] == 2)
        share = torch.softmax(torch.cat([10.0 * keys[b, calo] @ slots[b].T, background[b, calo]], -1), -1)
        torch.testing.assert_close(out[b], share[:, :-1].T @ X[b, calo, 10:12])
        torch.testing.assert_close(out[b].sum(0) + (share[:, -1:] * X[b, calo, 10:12]).sum(0), X[b, calo, 10:12].sum(0))


def test_slots_not_seeded_on_tracks_take_the_assigned_energy():
    torch.manual_seed(0)
    decoder = ParticleSetDecoder(DIM, NUM_CLASSES, SetDecoderConfig(num_slots=8, num_layers=2, num_heads=2, **LATE))
    X, mask = colliderml_inputs()
    out = decoder(torch.randn(2, X.shape[1], DIM), mask, X)
    energy = assigned_energy(out.hit_grouping, mask, X, 4).sum(-1) + 1e-3
    torch.testing.assert_close(out.hit_energy.sum(-1) + 1e-3, energy)
    _, pid, momentum, _ = out.predictions
    track = (out.seed_index >= 0) & (torch.gather(X[..., 0], 1, out.seed_index.clamp_min(0)) == 1)
    assert track.any() and (~track).any()
    mass = torch.softmax(pid[..., 1:], -1)[..., 1] * 0.7
    p = torch.sqrt(energy.square() - (mass * torch.tanh(energy / mass)).square())
    torch.testing.assert_close(momentum[..., 4][~track], energy.log()[~track])
    torch.testing.assert_close(momentum[..., 0][~track], (p.log() - torch.log(torch.cosh(momentum[..., 1])))[~track], rtol=1e-4, atol=1e-4)
    assert not torch.allclose(momentum[..., 4][track], energy.log()[track])


def test_energy_readout_trains_the_assignment_with_no_unused_parameters():
    torch.manual_seed(0)
    decoder = ParticleSetDecoder(DIM, NUM_CLASSES, SetDecoderConfig(num_slots=8, num_layers=2, num_heads=2, hit_grouping_chunk_size=3, **LATE))
    X, mask = colliderml_inputs()
    out = decoder(torch.randn(2, X.shape[1], DIM, requires_grad=True), mask, X)
    out.predictions[2][..., 4].sum().backward(retain_graph=True)
    for name in ("hit_grouping_hit", "hit_grouping_slot", "hit_grouping_background"):
        assert getattr(decoder, name).weight.grad.abs().sum() > 0, name
    assert decoder.assignment_log_scale.grad.abs() > 0 and decoder.energy_correction[-1].weight.grad.abs().sum() > 0
    out.predictions[2][..., 0].sum().backward(retain_graph=True)
    assert decoder.readout_log_mass.grad.abs().sum() > 0
    sum(p.square().mean() for p in out.predictions).backward()
    assert not hasattr(decoder, "hit_grouping_projection")
    assert [name for name, parameter in decoder.named_parameters() if parameter.grad is None] == []


def test_assignment_scale_is_bounded_smoothly():
    torch.manual_seed(0)
    decoder = ParticleSetDecoder(DIM, NUM_CLASSES, SetDecoderConfig(num_slots=8, num_layers=1, num_heads=2, **LATE))
    memory, slots = torch.randn(1, 5, DIM), torch.randn(1, 8, DIM)
    for raw, lo, hi in ((0.0, 9.99, 10.01), (5.0, 90.0, 100.0), (-5.0, 1.0, 1.2)):
        with torch.no_grad():
            decoder.assignment_log_scale.fill_(raw)
        decoder.assignment_log_scale.grad = None
        _, slot_keys, _ = decoder._late_grouping(slots, memory)
        scale = 10.0 * slot_keys.norm(dim=-1)
        assert ((scale > lo) & (scale < hi)).all(), (raw, scale)
        slot_keys.norm(dim=-1).sum().backward()
        assert decoder.assignment_log_scale.grad.abs() > 0


def test_readout_pt_is_smooth_and_below_energy():
    m = torch.tensor(0.7)
    log_e = torch.linspace(-12.0, 4.0, 801, requires_grad=True)
    e = log_e.exp()
    log_p = log_e + 0.5 * torch.log1p(-(m * torch.tanh(e / m) / e).square().clamp_max(1.0 - 1e-6))
    (grad,) = torch.autograd.grad(log_p.sum(), log_e)
    assert torch.isfinite(log_p).all() and (log_p < log_e).all() and torch.isfinite(grad).all()


def test_energy_weighting_reweights_hits_and_is_neutral_for_equal_energies():
    batch = grouping_batch()
    g = torch.Generator().manual_seed(1)
    grouping = (torch.randn(1, 5, 3, generator=g), torch.randn(1, 2, 3, generator=g), torch.randn(1, 5, 1, generator=g))
    matches = [(torch.tensor([0, 1]), torch.tensor([0, 1]))]
    plain = hit_grouping_loss(grouping, batch, matches, chunk_size=2)
    weighted = hit_grouping_loss(grouping, batch, matches, chunk_size=2, energy_weighted=True)
    assert not torch.isclose(plain, weighted)
    batch.X[0, 1:, 5] = 1.7
    torch.testing.assert_close(plain, hit_grouping_loss(grouping, batch, matches, chunk_size=2, energy_weighted=True))


def test_model_step_trains_late_grouping_end_to_end():
    from mlpf.model.training import model_step

    torch.manual_seed(0)
    model = MLPF(mlpf_config(hit_grouping_energy_weighted=True, **LATE))
    batch = model_batch()
    loss, losses, _, _, _, _ = model_step(batch, model, None, REGRESSION_WEIGHTS)
    assert torch.isfinite(loss) and losses["HitGrouping"] > 0
    loss.backward()
    assert model.set_decoder.energy_correction[0].weight.grad is not None
    p = unpack_predictions(model(batch.X, batch.mask).predictions)
    assert torch.isfinite(p["energy"]).all() and math.isfinite(float(p["pt"].sum()))


def test_detach_stops_the_energy_output_at_the_assignment_and_changes_no_value():
    X, mask = colliderml_inputs()
    memory = torch.randn(2, X.shape[1], DIM, generator=torch.Generator().manual_seed(1))
    decoders = []
    for detach in (False, True):
        torch.manual_seed(0)
        config = SetDecoderConfig(num_slots=8, num_layers=2, num_heads=2, hit_grouping_chunk_size=3, hit_energy_readout_detach=detach, **LATE)
        decoders.append(ParticleSetDecoder(DIM, NUM_CLASSES, config))
    a, b = (dec(memory, mask, X) for dec in decoders)
    assert all(torch.equal(p, q) for p, q in zip(a.predictions, b.predictions))
    for out, dec, reaches in ((a, decoders[0], True), (b, decoders[1], False)):
        (out.predictions[2][..., 4].sum() + out.predictions[2][..., 0].sum()).backward(retain_graph=True)
        for name in ("hit_grouping_hit", "hit_grouping_slot", "hit_grouping_background"):
            grad = getattr(dec, name).weight.grad
            assert (grad is not None and grad.abs().sum() > 0) == reaches, name
        assert dec.energy_correction[-1].weight.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="hit_energy_readout_detach requires hit_energy_readout"):
        SetDecoderConfig(query_init="input-conditioned", hit_grouping=True, hit_grouping_late=True, hit_energy_readout_detach=True)


def hand_grouping(background=-100.0):
    # slot 0 claims hits 1-3, slot 1 nothing certain; hit 4 goes to the background
    keys = torch.tensor([[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]])
    slots = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    return keys, slots, torch.tensor([[[background], [background], [background], [background], [100.0]]])


def test_dice_cost_is_zero_for_the_owning_slot():
    batch = grouping_batch()
    keys = torch.tensor([[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]])
    slots = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    background = torch.tensor([[[-100.0], [-100.0], [-100.0], [-100.0], [100.0]]])
    (cost,) = assignment_dice_costs((keys, slots, background), batch, chunk_size=2)
    assert cost.shape == (2, 2)
    assert cost[0, 0] < 1e-3 and cost[1, 1] < 1e-3 and cost[0, 1] > 0.99 and cost[1, 0] > 0.99
    # kinematics alone prefer the swapped pairing; a large Dice weight overrides them
    targets = {"cls_id": torch.tensor([[1, 2]]), "eta": torch.tensor([[0.0, 1.0]]), "sin_phi": torch.zeros(1, 2), "cos_phi": torch.ones(1, 2)}
    targets |= {"pt": torch.zeros(1, 2), "energy": torch.zeros(1, 2)}
    predictions = {"cls_binary": torch.zeros(1, 2, 2), "cls_id_onehot": torch.zeros(1, 2, NUM_CLASSES), "eta": torch.tensor([[1.0, 0.0]])}
    predictions |= {"sin_phi": torch.zeros(1, 2), "cos_phi": torch.ones(1, 2), "pt": torch.zeros(1, 2), "energy": torch.zeros(1, 2)}
    swapped = hungarian_match(targets, predictions, torch.ones(1, 2))
    assert swapped[0][0][swapped[0][1].argsort()].tolist() == [1, 0]
    steered = hungarian_match(targets, predictions, torch.ones(1, 2), extra_costs=[100.0 * cost])
    assert steered[0][0][steered[0][1].argsort()].tolist() == [0, 1]


def test_dice_loss_hand_numbers():
    batch = grouping_batch()
    matches = [(torch.tensor([0, 1]), torch.tensor([0, 1]))]
    loss = assignment_dice_loss(hand_grouping(), batch, matches, chunk_size=2)
    # slot 0 takes 2 + 1 + 4 GeV of which its target owns 3: Dice 0.6; slot 1 takes nothing of its target's 4 GeV
    assert float(loss) == pytest.approx(((1 - 0.6) + 1.0) / 2, abs=1e-4)
    assert float(assignment_dice_loss(hand_grouping(), batch, matches, chunk_size=5)) == pytest.approx(float(loss), abs=1e-6)
    batch.ytarget[0, :, 13] = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0])  # no target owns calorimeter energy
    assert float(assignment_dice_loss(hand_grouping(), batch, matches)) == 0.0


def test_dice_loss_reaches_the_assignment_and_the_model_step_reports_it():
    from mlpf.model.training import model_step

    keys, slots, background = (t.clone().requires_grad_() for t in hand_grouping(background=0.0))
    assignment_dice_loss((keys, slots, background), grouping_batch(), [(torch.tensor([0, 1]), torch.tensor([0, 1]))], chunk_size=2).backward()
    assert all(t.grad.abs().sum() > 0 for t in (keys, slots, background))
    batch = model_batch()
    losses = {}
    for weight in (0.0, 2.0):
        torch.manual_seed(0)
        model = MLPF(mlpf_config(hit_grouping_energy_weighted=True, hit_assignment_dice_cost=2.0, hit_grouping_dice_loss_weight=weight, **LATE))
        loss, losses[weight], _, _, _, _ = model_step(batch, model, None, REGRESSION_WEIGHTS)
        assert torch.isfinite(loss)
    assert "HitDice" not in losses[0.0] and 0 < float(losses[2.0]["HitDice"]) <= 2.0
    assert float(losses[2.0]["HitGrouping"]) == pytest.approx(float(losses[0.0]["HitGrouping"]), rel=1e-6)
    with pytest.raises(ValueError, match="require hit_grouping"):
        SetDecoderConfig(query_init="input-conditioned", hit_grouping_dice_loss_weight=2.0)
