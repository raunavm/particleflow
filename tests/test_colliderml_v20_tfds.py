import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import mlpf.heptfds.colliderml_utils.utils as U
from mlpf.data.colliderml.postprocessing import _batch_arrow_arrays
from mlpf.data.colliderml.v20_adapter import parse_splits

PU0_SPLITS = ["train:0-7679:90", "val:7680-8959:10", "report:8960-10239:10"]
L = U.V20_PU0_LAYOUT


def _rec(eid, rng, n_trk=2, n_cl=3, n_gen=4):
    """A record with the converter's keys and widths."""
    f32 = np.float32
    yt, yc = rng.random((n_trk, 14)).astype(f32), rng.random((n_cl, 14)).astype(f32)
    yt[:, 0] = rng.choice([211, 11, 13, 0], n_trk)
    yc[:, 0] = rng.choice([130, 22, 2112, 0], n_cl)
    gen = rng.random((n_gen, 10)).astype(f32)
    gen[:, 0] = rng.choice([211, 130, 22, 11, 13], n_gen)
    nj = int(rng.integers(0, 3))
    return {
        "event_id": np.int64(eid),
        "X_track": rng.random((n_trk, 17)).astype(f32),
        "X_cluster": rng.random((n_cl, 17)).astype(f32),
        "ytarget_track": yt,
        "ytarget_cluster": yc,
        "genmet": f32(rng.random() * 50),
        "genjet": rng.random((nj, 4)).astype(f32),
        "targetjet": rng.random((nj, 4)).astype(f32),
        "gen_target": gen,
        "genmet_withpu": f32(rng.random() * 50),
        "genjet_withpu": rng.random((nj, 4)).astype(f32),
        "pufrac_track": rng.random(n_trk).astype(f32),
        "pufrac_cluster": rng.random(n_cl).astype(f32),
    }


def _write(path, events, drop=(), row_group_size=2):
    cols = [(k, c) for k, c in _batch_arrow_arrays(events) if k not in drop]
    pq.write_table(pa.Table.from_arrays([c for _, c in cols], names=[k for k, _ in cols]), path, row_group_size=row_group_size)
    return path


def _events(ids, seed=0):
    rng = np.random.default_rng(seed)
    evs = [_rec(e, rng) for e in ids]
    evs[1] = _rec(ids[1], rng, n_trk=0, n_cl=0, n_gen=0)
    return evs


SIX = ("X", "ytarget", "ycand", "genmet", "genjets", "targetjets")


def _same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b)


def test_generate_examples_v20(tmp_path):
    files = [_write(tmp_path / "a.parquet", _events(range(10, 17))), _write(tmp_path / "b.parquet", _events(range(30, 33), seed=1))]
    old = list(U.generate_examples(files))
    new = list(U.generate_examples_v20(files))
    assert [k for k, _ in new] == [10, 12, 13, 14, 15, 16, 30, 32]
    assert len(old) == len(new)
    for (_, o), (k, n) in zip(old, new):
        assert all(_same(o[f], n[f]) for f in SIX)
        assert n["event_id"] == k and n["event_id"].dtype == np.int64


def test_gen_target(tmp_path):
    evs = _events(range(5))
    out = dict(U.generate_examples_v20([_write(tmp_path / "a.parquet", evs)]))
    for e in evs:
        if int(e["event_id"]) == 1:
            assert 1 not in out
            continue
        g = out[int(e["event_id"])]["gen_target"]
        assert g.dtype == np.float32 and g.shape == (len(e["gen_target"]), 10)
        assert g[:, 0].tolist() == [U.labels.index(int(c)) for c in e["gen_target"][:, 0]]
        assert np.array_equal(g[:, 1:], e["gen_target"][:, 1:])


def test_missing_gen_target(tmp_path):
    fn = _write(tmp_path / "a.parquet", _events(range(3)), drop=("gen_target",))
    with pytest.raises(KeyError, match="gen_target"):
        list(U.generate_examples_v20([fn]))


def _layout_files(tmp_path, specs=PU0_SPLITS, seed=0):
    """Event-id-only files, one per adapter shard, under shuffled names."""
    shards = parse_splits(specs)
    perm = np.random.default_rng(seed).permutation(len(shards))
    for (_, ids), k in zip(shards, perm):
        pq.write_table(pa.table({"event_id": pa.array(ids, pa.int64())}), tmp_path / f"train-{k:05d}-of-{len(shards):05d}.parquet")
    return U.v20_file_event_ids(tmp_path), shards


def _ids(file_ids, files):
    return np.sort(np.concatenate([file_ids[f] for f in files])).tolist()


def test_config_files(tmp_path):
    fid, shards = _layout_files(tmp_path)
    train = [ids for name, ids in shards if name == "train"]
    report = [ids for name, ids in shards if name == "report"]
    assert len(train) == 90 and len(report) == 10
    assert all(len(ids) == 128 for name, ids in shards if name != "train")
    assert U.v20_config_names(L) == [str(k) for k in range(1, 11)] + [f"r{j}" for j in range(1, 11)]
    for c in U.v20_config_names(L):
        m = U.v20_config_files(fid, L, c)
        assert set(m) == U.v20_expected_splits(c)
        if c.startswith("r"):
            assert _ids(fid, m["test"]) == report[int(c[1:]) - 1].tolist()
            continue
        k = int(c)
        assert len(m["train"]) == 9
        assert _ids(fid, m["train"]) == np.concatenate(train[(k - 1) * 9 : k * 9]).tolist()
        assert max(_ids(fid, m["train"])) < L["report"][0]
    assert _ids(fid, U.v20_config_files(fid, L, "1")["test"]) == list(range(7680, 8320))
    assert _ids(fid, U.v20_config_files(fid, L, "2")["val_read"]) == list(range(8320, 8960))


def _one(tmp_path, name, ids):
    pq.write_table(pa.table({"event_id": pa.array(ids, pa.int64())}), tmp_path / name)


@pytest.mark.parametrize(
    "bad,match",
    [
        (lambda d: _one(d, "x.parquet", np.arange(8310, 8330)), "straddle"),
        (lambda d: _one(d, "x.parquet", np.array([20000])), "outside"),
        (lambda d: _one(d, "x.parquet", np.array([3])), "more than once"),
        (lambda d: _one(d, "x.parquet", np.array([], np.int64)), "no events"),
    ],
)
def test_bad_files_raise(tmp_path, bad, match):
    _layout_files(tmp_path)
    bad(tmp_path)
    with pytest.raises(ValueError, match=match):
        U.v20_config_files(U.v20_file_event_ids(tmp_path), L, "3")


def test_incomplete_layouts_raise(tmp_path):
    fid, _ = _layout_files(tmp_path)
    by_first = {int(v.min()): f for f, v in fid.items()}
    drop = lambda *firsts: {f: v for f, v in fid.items() if int(v.min()) not in firsts}  # noqa: E731
    with pytest.raises(ValueError, match="train shards"):
        U.v20_config_files(drop(0), L, "5")
    with pytest.raises(ValueError, match="report shards"):
        U.v20_config_files(drop(8960), L, "r1")
    with pytest.raises(ValueError, match="config 1 test: empty"):
        U.v20_config_files(drop(*range(7680, 8320, 128)), L, "1")
    with pytest.raises(ValueError, match="config 2 val_read: empty"):
        U.v20_config_files(drop(*range(8320, 8960, 128)), L, "2")
    short = dict(fid)
    short[by_first[7680]] = np.arange(7680, 7807)
    with pytest.raises(ValueError, match="exactly"):
        U.v20_config_files(short, L, "1")
    with pytest.raises(ValueError, match="unknown config"):
        U.v20_config_files(fid, L, "r11")


def test_tfds_build(tmp_path):
    tfds = pytest.importorskip("tensorflow_datasets")
    from mlpf.heptfds.colliderml_pf.ttbar_pu0_v20 import CollidermlTtbarPu0V20Pf

    assert CollidermlTtbarPu0V20Pf.name == "colliderml_ttbar_pu0_v20_pf"
    manual = tmp_path / "manual"
    manual.mkdir()
    rng = np.random.default_rng(1)
    for k, (_, ids) in enumerate(s for s in parse_splits(PU0_SPLITS) if s[0] == "report"):
        _write(manual / f"shard-{(7 * k) % 10:02d}.parquet", [_rec(int(e), rng, n_trk=1, n_cl=1, n_gen=2) for e in ids], row_group_size=32)
    data_dir = tmp_path / "tfds"
    CollidermlTtbarPu0V20Pf(data_dir=str(data_dir), config="r2").download_and_prepare(
        download_config=tfds.download.DownloadConfig(manual_dir=str(manual))
    )
    version = str(CollidermlTtbarPu0V20Pf.VERSION)
    ds = tfds.builder_from_directory(str(data_dir / "colliderml_ttbar_pu0_v20_pf" / "r2" / version)).as_data_source(split="test")
    assert len(ds) == 128
    ids = sorted(int(ds[i]["event_id"]) for i in range(len(ds)))
    assert ids == list(range(9088, 9216))
    r = ds[0]
    assert r["gen_target"].shape == (2, 10) and r["X"].shape == (2, 17)
    assert set(r["gen_target"][:, 0].tolist()) <= {1.0, 2.0, 3.0, 4.0, 5.0}
