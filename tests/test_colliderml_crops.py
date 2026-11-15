import copy
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mlpf.conf import MLPFConfig, Y_FEATURES, CropConfig
from mlpf.model import crops
from mlpf.model.PFDataset import Collater, CropDataset, EpochKeySampler, ShardConsecutiveSampler, TFDSDataSource, gen_target_to_set
from mlpf.model.set_losses import hungarian_match
from mlpf.model.utils import unpack_predictions, unpack_target
from test_set_matching import loss, make_raw, make_targets
from test_set_seeding import build_decoder, calo_inputs, make_inputs

DS_NAME = "colliderml_ttbar_pu200_v22_pf_blocks"
ID_COL = 16  # a spare X column used here to tag elements with their row


def make_record(seed=0, n_trk=60, n_cell=400, n_gen=120, event_id=7, eta_max=5.0):
    """One synthetic event: tracks with perigee columns 6-9 (d0, z0, theta, qop), cells, gen_target linked to tracks."""
    rng = np.random.default_rng(seed)
    X = np.zeros((n_trk + n_cell, 17), dtype=np.float32)
    t_eta = rng.uniform(-3.5, 3.5, n_trk)
    t_phi = rng.uniform(-math.pi, math.pi, n_trk)
    t_pt = rng.uniform(0.3, 30.0, n_trk)
    t_q = rng.choice([-1.0, 1.0], n_trk)
    X[:n_trk, 0] = 1
    X[:n_trk, 1] = t_pt
    X[:n_trk, 2] = t_eta
    X[:n_trk, 3], X[:n_trk, 4] = np.sin(t_phi), np.cos(t_phi)
    X[:n_trk, 5] = t_pt * np.cosh(t_eta)
    X[:n_trk, 6] = rng.normal(0, 0.05, n_trk)
    X[:n_trk, 7] = rng.normal(0, 30.0, n_trk)
    X[:n_trk, 8] = 2 * np.arctan(np.exp(-t_eta))
    X[:n_trk, 9] = t_q / (t_pt * np.cosh(t_eta))
    c_eta = rng.uniform(-eta_max, eta_max, n_cell)
    c_phi = rng.uniform(-math.pi, math.pi, n_cell)
    X[n_trk:, 0] = 2
    X[n_trk:, 1] = rng.uniform(0.01, 5.0, n_cell)
    X[n_trk:, 2] = c_eta
    X[n_trk:, 3], X[n_trk:, 4] = np.sin(c_phi), np.cos(c_phi)
    X[n_trk:, 5] = X[n_trk:, 1] * np.cosh(c_eta)
    X[:, ID_COL] = np.arange(len(X))

    ytarget = np.zeros((len(X), 14), dtype=np.float32)
    rows = rng.choice(len(X), size=n_gen // 2, replace=False)
    ytarget[rows, 0] = rng.integers(1, 6, len(rows))
    ytarget[rows, 2] = rng.uniform(0.5, 10.0, len(rows))
    ytarget[rows, 6] = ytarget[rows, 2] * 1.5
    ytarget[rows, 13] = np.arange(1, len(rows) + 1)

    cls = rng.integers(1, 6, n_gen)
    charge = np.where(np.isin(cls, [1, 4, 5]), rng.choice([-1.0, 1.0], n_gen), 0.0)
    g_eta = rng.uniform(-eta_max, eta_max, n_gen)
    g_phi = rng.uniform(-math.pi, math.pi, n_gen)
    g_pt = rng.uniform(0.2, 40.0, n_gen)
    gen = np.zeros((n_gen, 10), dtype=np.float32)
    gen[:, 0], gen[:, 1], gen[:, 2], gen[:, 3] = cls, charge, g_pt, g_eta
    gen[:, 4], gen[:, 5] = np.sin(g_phi), np.cos(g_phi)
    gen[:, 6] = g_pt * np.cosh(g_eta)
    gen[:, 7] = rng.integers(0, 2, n_gen)
    gen[:, 8] = rng.uniform(0, 1, n_gen)
    gen[:, 9] = rng.uniform(0, 5, n_gen)
    gen_track = np.full(n_gen, -1, dtype=np.int32)
    charged = np.flatnonzero(charge != 0)
    linked = charged[: min(len(charged) * 2 // 3, n_trk)]
    gen_track[linked] = rng.choice(n_trk, size=len(linked), replace=False)

    return {
        "X": X,
        "ytarget": ytarget,
        "ycand": np.zeros_like(ytarget),
        "genmet": np.float32(rng.uniform(0, 50)),
        "genjets": rng.uniform(1, 50, (5, 4)).astype(np.float32),
        "targetjets": rng.uniform(1, 50, (5, 4)).astype(np.float32),
        "event_id": np.int64(event_id),
        "gen_target": gen,
        "gen_target_track": gen_track,
    }


class _Features:
    def deserialize_example_np(self, record, decoders=None):
        return copy.deepcopy(record)


class _TFDS:
    def __init__(self, records):
        self.data_source = SimpleNamespace(__getitems__=lambda items: [records[i] for i in items])
        self.dataset_info = SimpleNamespace(name=DS_NAME, features=_Features(), config_name="e03200")
        self.decoders = None
        self.n = len(records)

    def __len__(self):
        return self.n


def make_concat(n_events_per_shard=(3, 3), pad_to_multiple=None, first_id=3200):
    shards, eid = [], first_id
    for n in n_events_per_shard:
        recs = [make_record(seed=eid, n_trk=20, n_cell=120, n_gen=40, event_id=eid + i) for i in range(n)]
        eid += n
        shards.append(
            TFDSDataSource(
                _TFDS(recs),
                sort=False,
                pad_to_multiple=pad_to_multiple,
                feature_dim=17,
                build_target_set=True,
                set_target_source="gen_target",
                raw=True,
            )
        )
    return torch.utils.data.ConcatDataset(shards)


def test_whole_event_crop_equals_the_uncropped_record():
    rec = make_record(seed=1, eta_max=2.4)
    src = TFDSDataSource(_TFDS([rec]), sort=False, pad_to_multiple=64, feature_dim=17, build_target_set=True, set_target_source="gen_target")
    plain = src.finalize(copy.deepcopy(rec))
    whole = src.finalize(crops.crop_record(copy.deepcopy(rec), None, 0.2))
    interior = whole.pop("elem_interior")
    assert set(plain) == set(whole) and all(np.array_equal(plain[k], whole[k]) for k in plain)
    n = len(rec["X"])
    assert interior.dtype == bool and len(interior) == len(plain["X"]) and interior[:n].all() and not interior[n:].any()


def test_interiors_partition_elements_and_targets():
    rec = make_record(seed=2)
    rec["ytarget_set"] = gen_target_to_set(rec["gen_target"])
    pos = crops.positions(rec)
    seen_elem, seen_tgt = [], []
    for c in range(crops.n_crops()):
        r = crops.crop_record(rec, c, 0.2, pos=pos)
        seen_elem.append(r["X"][r["elem_interior"], ID_COL])
        seen_tgt.append(r["ytarget_set"][:, Y_FEATURES.index("particle_number")])
        assert len(r["gen_target"]) == len(r["ytarget_set"]) == len(r["gen_target_track"])
    assert np.array_equal(np.sort(np.concatenate(seen_elem)), np.arange(len(rec["X"])))
    assert np.array_equal(np.sort(np.concatenate(seen_tgt)), np.arange(1, len(rec["gen_target"]) + 1))


def test_phi_wrap_and_open_outer_eta_bins():
    eps = 1e-6
    eta = np.array([0.0, 0.0, 0.0])
    phi = np.array([math.pi - eps, -math.pi + eps, 0.0])
    lo, hi = 2 * crops.NPHI, 2 * crops.NPHI + crops.NPHI - 1
    assert crops.interior_id(eta, phi).tolist() == [hi, lo, 2 * crops.NPHI + 2]
    for c in (lo, hi):
        assert crops.in_crop(eta, phi, c, 0.2).tolist() == [True, True, False]
        assert crops.in_crop(eta, phi, c, 0.0).tolist() == [c == hi, c == lo, False]
    far = np.array([-1e3, -7.0, -4.5, 4.01, 6.0, 1e3])
    assert (crops.interior_id(far, np.zeros_like(far)) // crops.NPHI).tolist() == [0, 0, 0, 4, 4, 4]


def test_linked_target_and_its_track_share_an_interior():
    rec = make_record(seed=9)
    epos, tpos = crops.positions(rec)
    gt = rec["gen_target_track"].astype(np.int64)
    linked = np.flatnonzero((gt >= 0) & (rec["gen_target"][:, 1] != 0))
    track_rows = np.flatnonzero(rec["X"][:, 0] == 1)
    assert len(linked) > 10
    assert np.array_equal(crops.interior_id(*tpos)[linked], crops.interior_id(*epos)[track_rows[gt[linked]]])
    assert not np.allclose(epos[0][track_rows], rec["X"][track_rows, 2])


def test_crop_keys_carry_the_epoch_and_validation_crops_are_fixed():
    concat = make_concat((2, 2))
    train = CropDataset(concat, "train", CropConfig(enabled=True), seed=11)
    sampler = EpochKeySampler(ShardConsecutiveSampler(concat, shuffle=True, seed=11), len(concat))
    for epoch in (0, 2):
        sampler.set_epoch(epoch)
        for key in sampler:
            recs = train[key]
            eid = int(recs[0]["event_id"])
            assert [int(r["crop_id"]) for r in recs] == crops.choose_crops(11, epoch, eid, 4, 20)
    valid = CropDataset(concat, "valid", CropConfig(enabled=True, val_seed=4242))
    for key in (0, 1, 2 * len(concat), 5 * len(concat) + 1):
        recs = valid[key]
        assert [int(r["crop_id"]) for r in recs] == crops.choose_crops(4242, 0, int(recs[0]["event_id"]), 4, 20)


def test_crop_batches_hold_the_targets_of_their_interior():
    concat = make_concat((2,), pad_to_multiple=32)
    ds = CropDataset(concat, "train", CropConfig(enabled=True), seed=1)
    collate = Collater(["X", "ytarget", "ytarget_set", "elem_interior"], ["genmet", "source_id", "input_type_id", "event_id", "crop_id"])
    batch = collate([ds[0], ds[1]]).to("cpu")
    assert batch.X.shape[0] == 8 and batch.X.shape[1] % 32 == 0
    assert batch.event_id.tolist() == [3200] * 4 + [3201] * 4
    assert not (batch.elem_interior.bool() & ~batch.mask).any()
    for i in range(8):
        _, tpos = crops.positions(concat[i // 4])
        want = np.flatnonzero(crops.interior_id(*tpos) == int(batch.crop_id[i])) + 1
        got = batch.ytarget_set[i, batch.target_mask[i], Y_FEATURES.index("particle_number")].numpy()
        assert np.array_equal(got, want)


def test_unmatched_guard_slots_are_left_out_of_the_presence_loss():
    targets = make_targets(num_targets=1, seed=5)
    seed_index = torch.tensor([[0, 1, 2, 3, -1, -1]])
    batch = SimpleNamespace(target_mask=targets[..., 0] != 0, elem_interior=torch.tensor([[1.0, 1.0, 0.0, 0.0, 1.0]]))
    raw = make_raw(1, 6)
    matched = int(hungarian_match(unpack_target(targets, None), unpack_predictions(raw), batch.target_mask)[0][0])
    total, _, _ = loss(targets, raw, batch, seed_index=seed_index)
    total.backward()
    presence_grad = raw[0].grad[0].abs().sum(-1)
    for slot in range(6):
        assert (presence_grad[slot] == 0) == (slot in (2, 3) and slot != matched)
    # all interior is the same loss as no crop
    results = []
    for interior in (None, torch.ones(1, 5)):
        raw = make_raw(1, 6)
        total, _, _ = loss(targets, raw, SimpleNamespace(target_mask=batch.target_mask, elem_interior=interior), seed_index=seed_index)
        results.append(total)
    assert torch.equal(*results)


def test_seeds_rank_interior_first():
    decoder = build_decoder(num_slots=4, tracker_query_fraction=0.5)
    X, mask, memory = make_inputs(batch_size=1, num_elements=8, seed=4, num_tracks=2)
    X[0, 2:, 1] = 0.1
    X[0, 2:, 5] = torch.tensor([100.0, 90.0, 80.0, 1.0, 2.0, 3.0])
    interior = torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]])
    assert decoder(memory, mask, X).seed_index[0, 2:].tolist() == [2, 3]
    assert decoder(memory, mask, X, interior).seed_index[0, 2:].tolist() == [7, 6]
    assert torch.equal(decoder(memory, mask, X, torch.ones(1, 8)).seed_index, decoder(memory, mask, X).seed_index)


def test_anchors_in_a_crop_are_its_interior_and_guard_bins():
    cfg = CropConfig()
    decoder = build_decoder(num_slots=40, tracker_query_fraction=0.25, anchor_mode="fill_unseeded")
    decoder.crop_geometry = (cfg.eta_edges, cfg.nphi, cfg.guard)
    g = torch.Generator().manual_seed(14)
    X, mask, memory = calo_inputs(
        torch.rand(12, generator=g) * 2 - 1, torch.rand(12, generator=g) * 2 - 1.7, torch.rand(12, generator=g) * 4, num_tracks=3
    )
    eta, phi, _, interior = decoder._energy_anchors(X[0], mask[0] & (X[0, :, 0] == 2), torch.tensor(9))
    assert len(eta) == 1600  # 40 x 40 bins, guard 0.2
    assert torch.equal(interior, torch.from_numpy(crops.interior_id(eta.numpy(), phi.numpy()) == 9))
    out = decoder(memory, mask, X, torch.ones(1, 15), torch.tensor([9]))
    assert out.is_anchor[0, 15:].all() and not out.anchor_guard[0, 15:].any()  # interior bins rank first


def test_crop_config_validation():
    samples = {"physical": {"batch_size": 1, "samples": {DS_NAME: {"version": "1.2.0", "splits": ["e03200"]}}}}
    d = {
        "dataset": "colliderml",
        "data_dir": "/tmp",
        "model": {"type": "attention", "output_mode": "set", "attention": {"num_convs": 1, "num_heads": 2, "head_dim": 8, "attention_type": "math"}},
        "conv_type": "attention",
        "train_dataset": {"colliderml": samples},
        "valid_dataset": {"colliderml": samples},
        "set_target_source": "gen_target",
        "crop": {"enabled": True},
    }
    MLPFConfig.model_validate(d)
    with pytest.raises(ValueError, match="set_target_source='gen_target'"):
        MLPFConfig.model_validate(dict(d, set_target_source="element_rows"))
    with pytest.raises(ValueError, match="sort_data"):
        MLPFConfig.model_validate(dict(d, sort_data=True))
    with pytest.raises(ValueError, match="eta_edges"):
        CropConfig(eta_edges=(-1.0, 1.0))


def test_model_step_on_a_crop_batch():
    from mlpf.model.mlpf import MLPF
    from mlpf.model.training import model_step

    samples = {"physical": {"batch_size": 1, "samples": {DS_NAME: {"version": "1.2.0", "splits": ["e03200"]}}}}
    decoder = {"num_slots": 64, "num_layers": 2, "num_heads": 2, "query_init": "input-conditioned", "local_attention_radius": 0.4}
    decoder |= {"strict_seeding": True, "anchor_mode": "fill_unseeded", "allow_unmatched_targets": True, "presence_normalization": "num_slots"}
    config = MLPFConfig.model_validate(
        {
            "dataset": "colliderml",
            "data_dir": "/tmp",
            "model": {
                "type": "attention",
                "output_mode": "set",
                "attention": {"num_convs": 1, "num_heads": 2, "head_dim": 8, "attention_type": "math"},
                "set_decoder": decoder,
            },
            "conv_type": "attention",
            "train_dataset": {"colliderml": samples},
            "set_target_source": "gen_target",
            "crop": {"enabled": True},
        }
    )
    torch.manual_seed(0)
    model = MLPF(config)
    ds = CropDataset(make_concat((1,)), "train", config.crop, seed=1)
    collate = Collater(["X", "ytarget", "ytarget_set", "elem_interior"], ["genmet", "source_id", "input_type_id", "event_id", "crop_id"])
    batch = collate([ds[0]])
    total, losses, _, raw, _, _ = model_step(batch, model, None, {k: 1.0 for k in ("pt", "eta", "sin_phi", "cos_phi", "energy")})
    assert torch.isfinite(total) and losses["AnchorSlots"] > 0 and raw.anchor_guard is not None
    total.backward()
    assert model.set_decoder.anchor_embedding.grad.abs().sum() > 0
