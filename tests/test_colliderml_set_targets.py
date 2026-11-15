import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mlpf.conf import Y_FEATURES, MLPFConfig
from mlpf.model.PFDataset import TFDSDataSource, align_particle_numbers, gen_target_to_set
from mlpf.model.utils import unpack_target

DS_NAME = "colliderml_ttbar_pu0_v20_pf"


def make_record(seed=0, n_trk=3, n_calo=6, n_gen=4):
    rng = np.random.default_rng(seed)
    n = n_trk + n_calo
    X = rng.random((n, 17)).astype(np.float32)
    X[:, 0] = 2.0
    X[:n_trk, 0] = 1.0
    ytarget = np.zeros((n, 14), np.float32)
    ytarget[:n_gen, 0] = 1.0
    ytarget[:n_gen, 13] = np.arange(1, n_gen + 1)
    gen = rng.random((n_gen, 10)).astype(np.float32) + 0.1
    gen[:, 0] = rng.integers(1, 6, n_gen)
    gen[:, 7] = rng.integers(0, 2, n_gen)
    return {
        "X": X,
        "ytarget": ytarget,
        "ycand": np.zeros_like(ytarget),
        "genmet": np.float32(1.0),
        "event_id": np.int64(seed),
        "gen_target": gen,
        "gen_target_track": np.full(n_gen, -1, np.int32),
    }


class _Features:
    def deserialize_example_np(self, record, decoders=None):
        return copy.deepcopy(record)


class _TFDS:
    def __init__(self, records):
        self.data_source = SimpleNamespace(__getitems__=lambda items: [records[i] for i in items])
        self.dataset_info = SimpleNamespace(name=DS_NAME, features=_Features(), config_name="1")
        self.decoders = None
        self.n = len(records)

    def __len__(self):
        return self.n


def source(records, **kwargs):
    return TFDSDataSource(_TFDS(records), sort=False, pad_to_multiple=None, feature_dim=17, build_target_set=True, **kwargs)


def test_gen_target_set_columns_round_trip_through_unpack_target():
    gen = make_record(seed=12)["gen_target"]
    ys = gen_target_to_set(gen)
    assert ys.shape == (len(gen), len(Y_FEATURES)) and ys.dtype == np.float32
    t = unpack_target(torch.as_tensor(ys)[None], None)
    g = torch.as_tensor(gen)
    assert torch.equal(t["cls_id"][0], g[:, 0].long())
    assert torch.allclose(t["pt"][0], torch.log(g[:, 2]))
    assert torch.allclose(t["energy"][0], torch.log(g[:, 6]))
    for name, col in (("eta", 3), ("sin_phi", 4), ("cos_phi", 5), ("gp_to_track", 8), ("gp_to_cluster", 9)):
        assert torch.equal(t[name][0], g[:, col]), name
    assert torch.equal(t["ispu"][0], (g[:, 7] == 1).float())
    assert torch.equal(t["jet_idx"][0], -torch.ones(len(gen)))
    assert torch.equal(t["particle_number"][0], torch.arange(1, len(gen) + 1, dtype=torch.float32))


def test_loader_builds_set_targets_from_gen_target():
    rec = make_record(n_gen=5)
    got = source([rec], set_target_source="gen_target")[0]
    np.testing.assert_array_equal(got["ytarget_set"], gen_target_to_set(rec["gen_target"]))
    rows = source([rec])[0]["ytarget_set"]
    assert len(rows) == 5 and not np.array_equal(rows, got["ytarget_set"])


def test_particle_numbers_are_aligned_to_gen_target_rows():
    # gen_target rows 0-3; the allocator dropped row 1, so the kept particles 1, 2, 3 are gen_target rows 0, 2, 3
    gen = np.zeros((4, 10), np.float32)
    gen[:, 8] = [1.0, 0.0, 0.0, 1.0]
    gen[:, 9] = [0.0, 0.3, 2.5, 4.0]
    y = np.zeros((6, 14), np.float32)
    y[:, 13] = [1, 2, 3, 2, 3, 0]
    y[[0, 1, 2], 0] = 1.0
    y[[0, 1, 2], 10] = [1.0, 0.0, 1.0]
    y[[0, 1, 2], 11] = [0.0, 2.5, 4.0]
    rec = {"gen_target": gen, "ytarget": y, "event_id": 7}
    align_particle_numbers(rec)
    assert rec["ytarget"][:, 13].tolist() == [1, 3, 4, 3, 4, 0]
    rec["ytarget"] = y.copy()
    rec["ytarget"][2, 11] = 9.0
    with pytest.raises(ValueError, match="event 7"):
        align_particle_numbers(rec)


def _pf_config(sample="colliderml_ttbar_pu0_v20_pf", **top):
    samples = {"physical": {"batch_size": 1, "samples": {sample: {"version": "1.2.0", "splits": ["1"]}}}}
    return MLPFConfig.model_validate(
        {
            "dataset": "colliderml",
            "data_dir": "/tmp",
            "model": {
                "type": "attention",
                "output_mode": "set",
                "attention": {"num_convs": 1, "num_heads": 2, "head_dim": 8, "attention_type": "math"},
            },
            "conv_type": "attention",
            "train_dataset": {"colliderml": samples},
            "valid_dataset": {"colliderml": samples},
            "set_target_source": "gen_target",
            **top,
        }
    )


def test_set_output_accepted_on_v20_pf_datasets_only():
    _pf_config()
    _pf_config(sample="colliderml_ttbar_pu200_v22_pf_blocks")
    with pytest.raises(ValueError, match="ColliderML pf datasets"):
        _pf_config(sample="colliderml_ttbar_nopu_pf")
    with pytest.raises(ValueError, match="ColliderML pf datasets"):
        _pf_config(test_dataset={"colliderml_ttbar_nopu_pf": {"version": "1.5.0"}})
