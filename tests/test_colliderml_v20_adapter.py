import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mlpf.data.colliderml.make_test_fixture import make_fixture
from mlpf.data.colliderml.postprocessing import process_one_file
from mlpf.data.colliderml.reader import shard_paths
from mlpf.data.colliderml.truth import DEFAULT_CALIBRATION, calibration_factors
from mlpf.data.colliderml.v20_adapter import convert

N_EVENTS = 12


def _write(root, group_obj, first, last, table, campaign="hard_scatter"):
    d = root / group_obj
    d.mkdir(parents=True, exist_ok=True)
    tag = "v1" if group_obj == "truth/particles" else "v20"
    pq.write_table(table, d / f"{campaign}.ttbar.{tag}.{group_obj.replace('/', '.')}.events{first}-{last}.parquet")


def _v20_from_fixture(tmp_path):
    """The Release 1 fixture written as v20 files, with tracks shuffled."""
    make_fixture(tmp_path / "r1", n_events=N_EVENTS)
    r1 = {
        o: pa.concat_tables([pq.read_table(p) for p in shard_paths(tmp_path / "r1", "ttbar_pu0", o)])
        for o in ("particles", "tracks", "calo_hits", "tracker_hits")
    }
    rng = np.random.default_rng(7)
    v20 = tmp_path / "v20" / "parquet"
    truth = {"hits": [], "clusters": []}
    halves = [(0, N_EVENTS // 2 - 1), (N_EVENTS // 2, N_EVENTS - 1)]
    for first, last in halves:
        sel = pa.array(np.arange(first, last + 1))
        part = {o: t.take(sel) for o, t in r1.items()}
        _write(v20, "truth/particles", first, last, part["particles"])
        trk = part["tracks"]
        _write(v20, "reco/tracks", first, last, trk.take(pa.array(rng.permutation(trk.num_rows))))
        c = part["calo_hits"]
        rows = {k: [] for k in ("event_id", "detector", "cell_id", "energy", "x", "y", "z", "time", "contrib_particle_ids", "contrib_energies")}
        cl = {k: [] for k in ("event_id", "cluster_id", "energy", "x", "y", "z", "cell_ids")}
        for i in range(c.num_rows):
            ev = {k: c[k][i].as_py() for k in c.column_names}
            det = np.asarray(ev["detector"])
            e = np.asarray(ev["total_energy"], np.float64) * calibration_factors(det, DEFAULT_CALIBRATION)
            e[::9] = 0.0
            cid = (np.arange(len(det)) * 7 + 1000 * ev["event_id"] + 3).astype(np.uint64)
            for k in ("event_id", "detector", "x", "y", "z", "contrib_particle_ids", "contrib_energies"):
                rows[k].append(ev[k])
            rows["cell_id"].append(cid.tolist())
            rows["energy"].append(e.astype(np.float32).tolist())
            rows["time"].append([0.0] * len(det))
            # hit 0 is listed by two clusters
            lab = rng.integers(-1, 5, size=len(det))
            lab[rng.random(len(det)) < 0.2] = -1
            groups = [cid[lab == k].tolist() for k in range(5)]
            groups = [g for g in groups if g]
            if len(groups) >= 2 and len(cid):
                groups[1] = groups[1] + [int(cid[0])]
            cl_e = [float(e[np.isin(cid, g)].sum()) for g in groups]
            truth["hits"].append((ev["event_id"], cid, e))
            truth["clusters"].append((ev["event_id"], groups, cl_e))
            cl["event_id"].append(ev["event_id"])
            cl["cluster_id"].append(list(range(len(groups))))
            cl["energy"].append(cl_e)
            for k in ("x", "y", "z"):
                cl[k].append([0.0] * len(groups))
            cl["cell_ids"].append(groups)
        types = {
            "event_id": pa.uint32(),
            "detector": pa.list_(pa.uint8()),
            "cell_id": pa.list_(pa.uint64()),
            "energy": pa.list_(pa.float32()),
            "x": pa.list_(pa.float32()),
            "y": pa.list_(pa.float32()),
            "z": pa.list_(pa.float32()),
            "time": pa.list_(pa.float32()),
            "contrib_particle_ids": pa.list_(pa.list_(pa.uint64())),
            "contrib_energies": pa.list_(pa.list_(pa.float32())),
        }
        _write(v20, "reco/calo_cells", first, last, pa.table({k: pa.array(v, type=types[k]) for k, v in rows.items()}))
        cl_types = {
            "event_id": pa.uint32(),
            "cluster_id": pa.list_(pa.uint32()),
            "energy": pa.list_(pa.float32()),
            "x": pa.list_(pa.float32()),
            "y": pa.list_(pa.float32()),
            "z": pa.list_(pa.float32()),
            "cell_ids": pa.list_(pa.list_(pa.uint64())),
        }
        _write(v20, "reco/calo_clusters", first, last, pa.table({k: pa.array(v, type=cl_types[k]) for k, v in cl.items()}))
        th = part["tracker_hits"]
        pids = [[[] if j % 5 == 4 else [p] for j, p in enumerate(th["particle_id"][i].as_py())] for i in range(th.num_rows)]
        th = th.drop_columns(["particle_id"]).append_column("particle_ids", pa.array(pids, type=pa.list_(pa.list_(pa.uint64()))))
        _write(v20, "reco/tracker_hits", first, last, th)
    return v20, r1, truth


SPLITS = ["train:0-5:2", "val:6-8:1", "report:9-11:1"]


@pytest.fixture(scope="module")
def adapted(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("v20")
    v20, r1, truth = _v20_from_fixture(tmp)
    report = convert(v20, tmp / "src", "ttbar_pu0", SPLITS)
    return tmp, r1, truth, report


def _read(tmp, obj):
    return pa.concat_tables([pq.read_table(p) for p in shard_paths(tmp / "src", "ttbar_pu0", obj)])


def test_splits(adapted):
    tmp, _, _, report = adapted
    assert [(s["split"], s["first"], s["last"]) for s in report["shards"]] == [("train", 0, 2), ("train", 3, 5), ("val", 6, 8), ("report", 9, 11)]
    for obj in ("particles", "tracks", "calo_hits", "tracker_hits"):
        paths = shard_paths(tmp / "src", "ttbar_pu0", obj)
        assert [p.name for p in paths] == [f"train-{k:05d}-of-00004.parquet" for k in range(4)]
        assert np.concatenate([pq.read_table(p)["event_id"].to_numpy() for p in paths]).tolist() == list(range(N_EVENTS))


def test_tracks(adapted):
    tmp, r1, _, _ = adapted
    got = _read(tmp, "tracks")
    for k in ("d0", "qop", "hit_ids", "majority_particle_id"):
        assert got[k].to_pylist() == r1["tracks"][k].to_pylist()


def test_calo_columns(adapted):
    tmp, _, truth, report = adapted
    c = _read(tmp, "calo_hits")
    n_two = 0
    for i, (eid, cid, e) in enumerate(truth["hits"]):
        det = np.asarray(c["detector"][i].as_py())
        tot = np.asarray(c["total_energy"][i].as_py(), np.float64)
        np.testing.assert_allclose(tot * calibration_factors(det, DEFAULT_CALIBRATION), e, rtol=1e-6)
        lab = np.asarray(c["pandora_cluster"][i].as_py())
        _, groups, cl_e = truth["clusters"][i]
        expect = np.full(len(cid), -1)
        for k, g in enumerate(groups):
            for m in g:
                j = int(np.flatnonzero(cid == m)[0])
                if expect[j] == -1 or cl_e[k] > cl_e[expect[j]]:
                    expect[j] = k
        n_two += sum(len(g) for g in groups) - len({m for g in groups for m in g})
        assert lab.tolist() == expect.tolist()
    assert report["calo"]["hits_in_two_clusters"] == n_two > 0
    assert report["calo"]["cluster_hits_missing"] == 0


def test_tracker_particle_id(adapted):
    tmp, r1, _, _ = adapted
    got = _read(tmp, "tracker_hits")
    for i in range(N_EVENTS):
        orig = r1["tracker_hits"]["particle_id"][i].as_py()
        expect = [-1 if j % 5 == 4 else p for j, p in enumerate(orig)]
        assert got["particle_id"][i].as_py() == expect


def test_missing_cluster_file(tmp_path, adapted):
    tmp, _, _, _ = adapted
    v20, _, _ = _v20_from_fixture(tmp_path)
    gone = sorted((v20 / "reco/calo_clusters").glob("*.parquet"))[-1]
    first, last = (int(x) for x in gone.name.split(".events")[1].split(".")[0].split("-"))
    gone.unlink()
    with pytest.raises(RuntimeError, match="no-pandora-clusters"):
        convert(v20, tmp_path / "src_fail", "ttbar_pu0", SPLITS)
    shutil.rmtree(tmp_path / "src_fail")
    report = convert(v20, tmp_path / "src", "ttbar_pu0", SPLITS, no_pandora_clusters=True)
    assert report["calo"]["events_without_clusters"] == last - first + 1 > 0
    got = pa.concat_tables([pq.read_table(p) for p in shard_paths(tmp_path / "src", "ttbar_pu0", "calo_hits")])
    ref = _read(tmp, "calo_hits")
    assert got["event_id"].to_pylist() == ref["event_id"].to_pylist()
    for i, eid in enumerate(got["event_id"].to_pylist()):
        assert got["total_energy"][i].as_py() == ref["total_energy"][i].as_py()
        lab = got["pandora_cluster"][i].as_py()
        assert lab == ([-1] * len(lab) if first <= eid <= last else ref["pandora_cluster"][i].as_py())


ALGORITHMS = ("clue", "identity", "pandora")


@pytest.fixture(scope="module")
def converted(adapted):
    tmp = adapted[0]
    paths = [shard_paths(tmp / "src", "ttbar_pu0", o)[0] for o in ("particles", "tracks", "calo_hits", "tracker_hits")]
    out = {}
    for algo in ALGORITHMS:
        ofn = tmp / f"out_{algo}" / paths[0].name
        ofn.parent.mkdir()
        process_one_file(*paths, ofn, algorithm=algo)
        out[algo] = pq.read_table(ofn)
    return tmp, paths, out


def _calibrated_energy(calo, i):
    return np.asarray(calo["total_energy"][i].as_py(), np.float64) * calibration_factors(np.asarray(calo["detector"][i].as_py()), DEFAULT_CALIBRATION)


def test_identity_clusters(converted):
    _, paths, out = converted
    calo = pq.read_table(paths[2])
    for i in range(calo.num_rows):
        e = _calibrated_energy(calo, i)
        xi = np.asarray(out["identity"]["X_cluster"][i].as_py())
        assert len(xi) == (e > 0).sum()
        np.testing.assert_allclose(np.sort(xi[:, 5]), np.sort(e[e > 0]), rtol=1e-5)


def test_pandora_clusters(converted):
    _, paths, out = converted
    calo = pq.read_table(paths[2])
    for i in range(calo.num_rows):
        lab = np.asarray(calo["pandora_cluster"][i].as_py())
        e = _calibrated_energy(calo, i)
        xc = np.asarray(out["pandora"]["X_cluster"][i].as_py())
        assert len(xc) == len(np.unique(lab[lab >= 0]))
        np.testing.assert_allclose(xc[:, 5].sum(), e[lab >= 0].sum(), rtol=1e-5)
        h2c = np.asarray(out["pandora"]["hit_to_cluster"][i].as_py())
        assert ((h2c >= 0) == (lab >= 0)).all()
