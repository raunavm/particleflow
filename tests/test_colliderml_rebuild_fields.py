import awkward as ak
import numpy as np
import pyarrow.parquet as pq
import pytest

import mlpf.data.colliderml.postprocessing as pp
from mlpf.data.colliderml.make_test_fixture import make_fixture
from mlpf.data.colliderml.reader import shard_paths
from mlpf.data.colliderml.truth import own_hit_targets


def _no_jets(p4, min_pt, jetdef, with_indices=False):
    jets = ak.Array({"pt": np.zeros(0), "eta": np.zeros(0), "phi": np.zeros(0), "energy": np.zeros(0)})
    return (jets, ak.Array([[0]])[:0]) if with_indices else jets


@pytest.fixture(autouse=True)
def no_jets(monkeypatch):
    monkeypatch.setattr(pp, "compute_jets", _no_jets)


def _cid(layer, system, x=0):
    """A cell id in the ODD readout layout (system:8, barrel:3, module:4, stave:1, layer:6, ...)."""
    u = np.uint64
    return (u(x) << u(32)) | (u(3) << u(22)) | (u(layer) << u(16)) | (u(1) << u(15)) | u(system)


def _event(with_time_layer=True):
    R = lambda f: ak.Record({k: ak.from_iter(v) for k, v in f.items()})  # noqa: E731
    particles = {
        "particle_id": [1, 2, 3, 4],
        "pdg_id": [22, 22, 22, 211],
        "charge": [0.0, 0.0, 0.0, 1.0],
        "px": [200.0, 2.9, 0.0, 1.0],
        "py": [0.0, 0.3, 50.0, 0.0],
        "pz": [0.0, 0.0, 0.0, 10.0],
        "energy": [200.0, 2.92, 50.0, 10.1],
        "vertex_primary": [1] * 4,
        "primary": [True] * 4,
        "parent_id": [-1] * 4,
    }
    # hit 0: two targets 3:1; hit 1: one target; hit 2 (HCAL): half its energy from a particle not in the table;
    # hit 3: zero measured energy but a truth contribution
    calo = {
        "detector": [10, 10, 13, 10],
        "total_energy": [4.0, 2.0, 1.0, 0.0],
        "x": [1500.0, 0.0, 2000.0, 1400.0],
        "y": [0.0, 1500.0, 2000.0, 100.0],
        "z": [0.0, 0.0, 500.0, 0.0],
        "contrib_particle_ids": [[1, 2], [3], [1, 99], [3]],
        "contrib_energies": [[3.0, 1.0], [2.0], [0.5, 0.5], [0.1]],
        "contrib_times": [[0.0, 0.0], [0.0], [0.0, 0.0], [0.0]],
    }
    if with_time_layer:
        calo["cell_id"] = [int(_cid(layer, det, 11)) for layer, det in zip([1, 47, 20, 3], calo["detector"])]
        calo["time"] = [0.5, -0.2, 12.0, 0.0]
    tracks = {"d0": [0.01], "z0": [0.02], "phi": [0.3], "theta": [0.1], "qop": [0.1], "majority_particle_id": [4], "hit_ids": [[0, 1, 2]]}
    # hits 3 and 4 are on no track; 3 belongs to the pion, 4 to no particle
    tracker = {
        "particle_id": [4, 4, 4, 4, -1],
        "x": [30.0, 60.0, 90.0, 120.0, -40.0],
        "y": [0.0, 0.0, 0.0, 5.0, 7.0],
        "z": [300.0, 600.0, 900.0, 1200.0, -100.0],
        "time": [0.0, 0.0, 0.0, 0.4, 1.5],
        "detector": [0, 0, 0, 0, 2],
        "volume_id": [16, 16, 17, 17, 23],
        "layer_id": [2, 4, 2, 6, 8],
        "var_loc0": [2e-4, 2e-4, 3e-4, 4e-4, 5e-4],
        "var_loc1": [1e-4, 1e-4, 0.0, 0.0, 6e-4],
        "size_loc0": [1, 2, 3, 4, 5],
        "size_loc1": [1, 1, 1, 1, 2],
    }
    return R(particles), R(tracks), R(calo), R(tracker)


def test_cell_layer():
    u = np.array([_cid(0, 9), _cid(5, 10, 7), _cid(47, 13, 0xFFFFFFFF), _cid(35, 14, 3), _cid(63, 11)], np.uint64)
    for a in (u, u.view(np.int64)):
        assert pp.cell_layer(a).tolist() == [0, 5, 47, 35, 63]
    with pytest.raises(ValueError):
        pp.cell_layer(np.array([1.0]))


def test_time_layer_fractions_and_offtrack_hits():
    r = pp._event_record_one(7, *_event(), algorithm="identity")
    X = r["X_hit_calo"]
    assert X[:, 9].tolist() == pytest.approx([0.5, -0.2, 12.0, 0.0])
    assert X[:, 14].tolist() == [1, 47, 20, 3]
    gen = r["gen_target"]
    assert gen.shape[0] == 4
    got = list(zip(r["calo_frac_hit"].tolist(), r["calo_frac_target"].tolist(), r["calo_frac"].tolist()))
    assert [(h, g) for h, g, _ in got] == [(0, 0), (0, 1), (1, 2), (2, 0), (3, 2)]
    assert [f for _, _, f in got] == pytest.approx([0.75, 0.25, 1.0, 0.5, 1.0])
    assert r["calo_frac_hit"].dtype == np.int32 and r["calo_frac"].dtype == np.float32
    assert r["offtrack_hit"].tolist() == [3, 4] and r["offtrack_target"].tolist() == [3, -1]
    assert r["offtrack_own"].dtype == np.int32 and r["offtrack_own"].tolist() == [3, -1]
    np.testing.assert_array_equal(
        r["X_hit_offtrack"],
        np.float32([[120.0, 5.0, 1200.0, 0.4, 0.0, 17.0, 6.0, 4e-4, 0.0, 4.0, 1.0], [-40.0, 7.0, -100.0, 1.5, 2.0, 23.0, 8.0, 5e-4, 6e-4, 5.0, 2.0]]),
    )
    assert r["gen_target_track"].tolist() == [-1, -1, -1, 0]


@pytest.mark.parametrize("algorithm", ["identity", "clue"])
def test_time_and_layer_change_nothing_else(algorithm):
    a = pp._event_record_one(7, *_event(), algorithm=algorithm)
    b = pp._event_record_one(7, *_event(with_time_layer=False), algorithm=algorithm)
    assert list(a) == list(b)
    for k in a:
        va, vb = np.asarray(a[k]), np.asarray(b[k])
        if k == "X_hit_calo":
            assert (vb[:, [9, 14]] == 0).all() and (va[:, [9, 14]] != 0).any()
            va, vb = np.delete(va, [9, 14], axis=1), np.delete(vb, [9, 14], axis=1)
        assert va.dtype == vb.dtype and va.tobytes() == vb.tobytes(), k


def test_own_hit_targets():
    particles = {"particle_id": np.array([10, 20, 30, 2**63 + 7], np.uint64)}
    tracker = {"particle_id": np.array([20, 30, -1, 10, 20, (2**63 + 7) - 2**64], np.int64)}
    assert own_hit_targets(tracker, particles, np.array([1, 0, 3])).tolist() == [0, -1, -1, 1, 0, 2]
    assert own_hit_targets(tracker, particles, np.array([], np.int64)).tolist() == [-1] * 6


def test_hits_view_keeps_zero_time_and_layer(tmp_path):
    from mlpf.data.colliderml.postprocessing import _batch_arrow_arrays
    from mlpf.heptfds.colliderml_hits.utils import generate_examples
    import pyarrow as pa

    cols = _batch_arrow_arrays([pp._event_record_one(7, *_event(), algorithm="identity")])
    fn = tmp_path / "a.parquet"
    pq.write_table(pa.Table.from_arrays([c for _, c in cols], names=[k for k, _ in cols]), fn)
    ((_, rec),) = list(generate_examples([fn]))
    calo = rec["X"][:, 0] == 2
    assert calo.sum() == 4 and (rec["X"][calo][:, [9, 14]] == 0).all()


def test_resume_reconverts_outputs_without_offtrack_tag(tmp_path):
    make_fixture(tmp_path / "src", n_events=6)
    paths = [shard_paths(tmp_path / "src", "ttbar_pu0", obj)[0] for obj in ("particles", "tracks", "calo_hits", "tracker_hits")]
    ofn = tmp_path / "out" / "shard.parquet"
    pp.process_one_file(*paths, ofn)
    assert pp._parquet_usable(ofn)
    table = pq.read_table(ofn)
    metadata = dict(table.schema.metadata)
    del metadata[b"mlpf.offtrack_features"]
    pq.write_table(table.replace_schema_metadata(metadata), ofn)
    assert not pp._parquet_usable(ofn)


def test_adapter_output_without_rebuild_columns_is_refused(tmp_path):
    import pyarrow as pa

    calo, tracker = tmp_path / "calo.parquet", tmp_path / "tracker.parquet"
    pq.write_table(pa.table({c: [[0]] for c in ("pandora_cluster", "time", "cell_id")}), calo)
    pq.write_table(pa.table({c: [[0]] for c in pp.OFFTRACK_FEATURES}), tracker)
    pp._check_rebuild_columns(calo, tracker)
    pq.write_table(pa.table({c: [[0]] for c in ("pandora_cluster", "cell_id")}), calo)
    with pytest.raises(KeyError, match="time"):
        pp._check_rebuild_columns(calo, tracker)
