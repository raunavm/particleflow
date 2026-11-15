import math

import pytest
import torch

from mlpf.model.input_features import ColliderMLInputFeatures


def _calo(e, x=1250.0, y=0.0, z=300.0, ehcal=False):
    r = math.hypot(x, y)
    eta, phi = math.asinh(z / r), math.atan2(y, x)
    return [
        2.0,
        e / math.cosh(eta),
        eta,
        math.sin(phi),
        math.cos(phi),
        e,
        x,
        y,
        z,
        0.0,
        0.0 if ehcal else e,
        e if ehcal else 0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
    ]


def _track():
    return [1.0, 5.0, 0.5, 0.0, 1.0, 5.6, 0.02, 30.0, 1.1, 0.1, 0.5, 0.001, 33.0, 12.0, 0.0, 0.0, 0.0]


@pytest.fixture(scope="module")
def f():
    return ColliderMLInputFeatures()


def test_shape_finite_padding_zero(f):
    X = torch.tensor([[_calo(0.05), _track(), [0.0] * 17]])
    out = f(X)
    assert out.shape == (1, 3, f.num_output_features)
    assert torch.isfinite(out).all()
    assert (out[0, 2] == 0).all()
    assert out[0, 0, 1] == 1 and out[0, 1, 0] == 1


def test_values_are_near_one(f):
    X = torch.tensor([[_calo(0.01), _calo(500.0, x=-1800.0, y=2600.0, z=3500.0, ehcal=True), _track()]])
    assert f(X).abs().max() < 5


def test_energy_is_visible_against_position(f):
    a, b, c = (f(torch.tensor([[v]]))[0, 0, : f.NUM_BASE] for v in (_calo(0.05), _calo(0.1), _calo(0.05, z=305.0)))
    assert (b - a).norm() > 10 * (c - a).norm()
    assert (b - a)[6] == pytest.approx(math.log(2) / 3, rel=1e-3)


def test_fourier_block_resolves_neighboring_cells(f):
    a, c = (f(torch.tensor([[v]]))[0, 0] for v in (_calo(0.05), _calo(0.05, z=305.0)))
    assert (c - a)[f.NUM_BASE :].abs().max() > 10 * (c - a)[9].abs()


def test_small_energy_differences_survive_the_bf16_cast(f):
    oa, ob = f(torch.tensor([[_calo(0.01)]])), f(torch.tensor([[_calo(0.0105)]]))
    assert oa.dtype == torch.float32
    assert ob.bfloat16()[0, 0, 6] != oa.bfloat16()[0, 0, 6]


def test_saturation_flags(f):
    out = f(torch.tensor([[_calo(0.7, ehcal=True), _calo(0.7), _calo(9.7), _calo(9.7, ehcal=True)]]))[0]
    assert out[:, 18].tolist() == [1.0, 0.0, 0.0, 1.0]
    assert out[:, 19].tolist() == [0.0, 0.0, 1.0, 0.0]


def test_model_flag_changes_only_the_input_embedding():
    from mlpf.conf import MLPFConfig
    from mlpf.model.mlpf import MLPF

    d = {
        "dataset": "colliderml",
        "data_dir": "/tmp",
        "model": {"type": "attention", "attention": {"num_convs": 1, "num_heads": 2, "head_dim": 8, "attention_type": "math"}},
        "conv_type": "attention",
    }
    raw = MLPF(MLPFConfig.model_validate(d))
    d["model"]["colliderml_input_features"] = True
    model = MLPF(MLPFConfig.model_validate(d))
    sa, sb = raw.state_dict(), model.state_dict()
    assert list(sa) == list(sb)
    different = [k for k in sa if sa[k].shape != sb[k].shape]
    assert different and all("nn0" in k and ".0." in k for k in different), different
    X = torch.tensor([[_calo(5.0), _calo(0.5, z=-200.0), _track()]])
    assert model._encode_inputs(X, mask=torch.ones(1, 3, dtype=torch.bool)).shape[:2] == (1, 3)
    d["dataset"] = "cld"
    with pytest.raises(ValueError, match="colliderml_input_features"):
        MLPFConfig.model_validate(d)


def test_entry_block_holds_the_entry_point_for_tracks_and_the_own_direction_for_calorimeter_elements():
    from mlpf.conf import Dataset
    from mlpf.model.calo_entry import CALO_ENTRY_GEOMETRY_MM, calo_entry_eta_phi

    tracks = []
    for z0, eta, omega in ((0.0, 0.3, 1.0 / 3000.0), (30.0, -0.8, -1.0 / 1500.0), (-50.0, 1.2, 1.0 / 800.0)):
        t = _track()
        t[2], t[7], t[10], t[11] = eta, z0, math.sinh(eta), omega
        tracks.append(t)
    X = torch.tensor([tracks + [_calo(5.0), _calo(0.5, z=-200.0)]])
    off, on = ColliderMLInputFeatures(), ColliderMLInputFeatures(track_entry=True)
    a, b = off(X), on(X)
    n = ColliderMLInputFeatures.NUM_ENTRY
    assert on.num_output_features == off.num_output_features + n == b.shape[-1] and torch.equal(b[..., :-n], a)
    phi0 = torch.atan2(X[0, :, 3], X[0, :, 4])
    eta_e, phi_e = calo_entry_eta_phi(phi0[:3], X[0, :3, 7], X[0, :3, 10], X[0, :3, 11], *CALO_ENTRY_GEOMETRY_MM[Dataset.COLLIDERML.value])
    block = b[0, :, -n:]
    assert torch.allclose(block[:3, 0], eta_e / 3.0, atol=1e-6) and torch.allclose(block[:3, 1], torch.sin(phi_e), atol=1e-6)
    assert (phi_e - phi0[:3]).abs().min() > 0.05
    assert torch.allclose(block[3:, 0], X[0, 3:, 2] / 3.0) and torch.allclose(block[3:, 1], X[0, 3:, 3])


def test_entry_features_need_the_scaled_inputs():
    from mlpf.conf import MLPFConfig

    d = {
        "dataset": "colliderml",
        "data_dir": "/tmp",
        "model": {
            "type": "attention",
            "attention": {"num_convs": 1, "num_heads": 2, "head_dim": 8, "attention_type": "math"},
            "colliderml_track_entry_features": True,
        },
        "conv_type": "attention",
    }
    with pytest.raises(ValueError, match="needs model.colliderml_input_features"):
        MLPFConfig.model_validate(d)
