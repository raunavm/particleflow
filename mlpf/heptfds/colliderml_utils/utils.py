# Shared ColliderML -> TFDS utilities. Mirrors mlpf/heptfds/edm4hep_utils/utils_pf.py but reads
# the MLPF-format parquet produced by mlpf/data/colliderml/postprocessing.py (a dict-like record
# of X_track / X_cluster / ytarget_track / ytarget_cluster / genmet / genjet / targetjet /
# event_id, plus the pileup fields genmet_withpu / genjet_withpu / pufrac_track / pufrac_cluster).
from pathlib import Path
from typing import List

import awkward as ak
import numpy as np
import pyarrow.parquet as pq

# workaround for 'ModuleNotFoundError: No module named importlib_resources'
try:
    import importlib_resources  # noqa
except Exception:
    import sys
    import importlib.resources

    sys.modules["importlib_resources"] = importlib.resources

from mlpf.conf import ParticleFeatures

NUM_SPLITS = 10
Y_FEATURES = ParticleFeatures.get_names()
labels = [0, 211, 130, 22, 11, 13]

# X feature widths: tracks and clusters are written as the 16-wide layout by postprocessing;
# we pad/truncate to the registered X_FEATURES[Dataset.COLLIDERML] width (currently 17),
# which matches the EDM4hep joint-layout convention (union of track/cluster feature orders).
N_X_FEATURES = 17
N_Y_FEATURES = len(Y_FEATURES)


def split_list(lst, x):
    sublist_size = len(lst) // x
    result = [lst[i * sublist_size : (i + 1) * sublist_size] for i in range(x - 1)]
    result.append(lst[(x - 1) * sublist_size :])
    return result


def split_sample(path, builder_config, num_splits=NUM_SPLITS, test_frac=0.9, gen_fn=None):
    if gen_fn is None:
        gen_fn = generate_examples
    files = sorted(list(path.glob("*.parquet")))
    print("Found {} files in {}".format(len(files), path))
    assert len(files) > 0
    idx_split = int(test_frac * len(files))
    files_train = files[:idx_split]
    files_test = files[idx_split:]
    assert len(files_train) > 0
    assert len(files_test) > 0

    split_index = int(builder_config.name) - 1
    files_train_split = split_list(files_train, num_splits)
    files_test_split = split_list(files_test, num_splits)
    assert len(files_train_split[split_index]) > 0
    assert len(files_test_split[split_index]) > 0

    return {
        "train": gen_fn(files_train_split[split_index]),
        "test": gen_fn(files_test_split[split_index]),
    }


def _empty_y(n, w):
    return np.zeros((0, w), dtype=np.float32) if n == 0 else None


def _normalize_matrix(mat, width):
    """Convert an ak/numpy variable-length 2D matrix representation to a padded float32
    numpy array of shape (n, width). Accepts (n, m) ndarrays and slices the columns to the
    desired width (or pads with zeros to reach it)."""
    # inputs are already numpy/awkward-converted; this helper just normalizes the second axis
    m = np.asarray(mat)
    if m.ndim == 1 and m.size == 0:
        return np.zeros((0, width), dtype=np.float32)
    if m.shape[1] < width:
        return np.pad(m, ((0, 0), (0, width - m.shape[1])), mode="constant")
    return m[:, :width]


def _remap_labels(col):
    """Class-label index remap; matches existing heptfds convention."""
    return np.array([labels.index(int(p)) if int(p) in labels else 0 for p in col], dtype=np.float32)


def _prepare_event(ret, iev):
    """X, ytarget, ycand, genmet, genjets and targetjets of event iev, or None for an empty event."""
    X_track = ak.to_numpy(ret["X_track"][iev])
    X_cluster = ak.to_numpy(ret["X_cluster"][iev])
    y_track = ak.to_numpy(ret["ytarget_track"][iev])
    y_cluster = ak.to_numpy(ret["ytarget_cluster"][iev])

    if (X_track.shape[0] == 0 and X_cluster.shape[0] == 0) or (y_track.shape[0] == 0 and y_cluster.shape[0] == 0):
        return None

    X_track = _normalize_matrix(X_track, N_X_FEATURES)
    X_cluster = _normalize_matrix(X_cluster, N_X_FEATURES)
    y_track = _normalize_matrix(y_track, N_Y_FEATURES)
    y_cluster = _normalize_matrix(y_cluster, N_Y_FEATURES)
    if y_track.shape[0] == 0:
        y_track = _empty_y(0, N_Y_FEATURES)
    if y_cluster.shape[0] == 0:
        y_cluster = _empty_y(0, N_Y_FEATURES)

    # option-typed parquet scalars come back from ak.to_numpy as shape-(1,) arrays; reshape
    # keeps this correct for both 0-d and option-typed readback
    genmet = float(np.asarray(ak.to_numpy(ret["genmet"][iev])).reshape(-1)[0])
    genjet = ak.to_numpy(ret["genjet"][iev])
    targetjet = ak.to_numpy(ret["targetjet"][iev])
    if genjet.size == 0:
        genjet = np.zeros((0, 4), dtype=np.float32)
    if targetjet.size == 0:
        targetjet = np.zeros((0, 4), dtype=np.float32)

    X = np.concatenate([X_track, X_cluster], axis=0)
    ytarget = np.concatenate([y_track, y_cluster], axis=0)
    ycand = np.zeros(ytarget.shape, dtype=np.float32)
    ytarget[:, 0] = _remap_labels(ytarget[:, 0])
    return {"X": X, "ytarget": ytarget, "ycand": ycand, "genmet": genmet, "genjets": genjet, "targetjets": targetjet}


def prepare_events(fn: Path):
    ret = ak.from_parquet(fn)
    nev = len(ret["event_id"])
    Xs = []
    ytargets = []
    ycands = []
    genmets = []
    genjets = []
    targetjets = []
    pileup = []  # event_id, pufrac, genmet_withpu, genjets_withpu
    for iev in range(nev):
        rec = _prepare_event(ret, iev)
        if rec is None:
            continue

        Xs.append(rec["X"])
        ytargets.append(rec["ytarget"])
        ycands.append(rec["ycand"])
        genmets.append(rec["genmet"])
        genjets.append(rec["genjets"])
        targetjets.append(rec["targetjets"])
        genjet_withpu = ak.to_numpy(ret["genjet_withpu"][iev]).reshape(-1, 4)
        pileup.append(
            {
                # the release event_id, unique over the whole release: the link back to the raw tables
                "event_id": np.int64(np.asarray(ak.to_numpy(ret["event_id"][iev])).reshape(-1)[0]),
                # truth pileup fraction of each element, in X order (tracks, then clusters)
                "pufrac": np.concatenate([ak.to_numpy(ret["pufrac_track"][iev]), ak.to_numpy(ret["pufrac_cluster"][iev])]).astype(np.float32),
                "genmet_withpu": float(np.asarray(ak.to_numpy(ret["genmet_withpu"][iev])).reshape(-1)[0]),
                "genjets_withpu": genjet_withpu.astype(np.float32),
            }
        )
    return Xs, ytargets, ycands, genmets, genjets, targetjets, pileup


def generate_examples(files: List[Path]):
    for fi in files:
        Xs, ytargets, ycands, genmets, genjets, targetjets, pileup = prepare_events(fi)
        for iev in range(len(Xs)):
            yield (
                str(fi) + "_" + str(iev),
                {
                    "X": Xs[iev].astype(np.float32),
                    "ytarget": ytargets[iev].astype(np.float32),
                    "ycand": ycands[iev].astype(np.float32),
                    "genmet": genmets[iev],
                    "genjets": genjets[iev].astype(np.float32),
                    "targetjets": targetjets[iev].astype(np.float32),
                    **pileup[iev],
                },
            )


# ColliderML v20: files go to splits by the event ids they hold, from the v20 adapter's
# --split train:0-7679:90 --split val:7680-8959:10 --split report:8960-10239:10.
# Configs 1-10 each take 1/10 of the train shards, config 1 also holds test and config 2 val_read.
# Configs r1-r10 hold one report shard each as test.
V20_PU0_LAYOUT = {"train": (0, 7679), "test": (7680, 8319), "val_read": (8320, 8959), "report": (8960, 10239), "n_report": 10}
V20_RANGES = ("train", "test", "val_read", "report")
N_V20_TRAIN_CONFIGS = 10
V20_COLUMNS = ["event_id", "X_track", "X_cluster", "ytarget_track", "ytarget_cluster", "genmet", "genjet", "targetjet", "gen_target"]


def _in_range(ids, entry):
    lo, hi = entry
    ids = np.asarray(ids, np.int64)
    return (ids >= lo) & (ids <= hi)


def v20_config_names(layout):
    return [str(k) for k in range(1, N_V20_TRAIN_CONFIGS + 1)] + [f"r{j}" for j in range(1, layout["n_report"] + 1)]


def v20_expected_splits(config):
    if config.startswith("r"):
        return {"test"}
    return {"train"} | ({"test"} if config == "1" else set()) | ({"val_read"} if config == "2" else set())


def v20_file_event_ids(path):
    """Event ids of every parquet file in path."""
    files = sorted(Path(path).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files in {path}")
    return {f: pq.read_table(f, columns=["event_id"])["event_id"].to_numpy().astype(np.int64) for f in files}


def _v20_range_of(ids, layout, name):
    if len(ids) == 0:
        raise ValueError(f"{name}: no events")
    inside = {r: _in_range(ids, layout[r]) for r in V20_RANGES}
    hit = [r for r, m in inside.items() if m.any()]
    if len(hit) != 1 or not inside[hit[0]].all():
        raise ValueError(f"{name}: event ids {ids.min()}-{ids.max()} straddle splits or are outside all of them")
    return hit[0]


def assert_v20_splits(config, ids_by_split, layout):
    """Check the event ids of one config's splits."""
    if set(ids_by_split) != v20_expected_splits(config):
        raise ValueError(f"config {config}: splits {sorted(ids_by_split)}, expected {sorted(v20_expected_splits(config))}")
    for split, ids in ids_by_split.items():
        ids = np.asarray(ids, np.int64)
        if len(ids) == 0:
            raise ValueError(f"config {config} {split}: empty")
        if len(np.unique(ids)) != len(ids):
            raise ValueError(f"config {config} {split}: duplicate event ids")
        entry = layout["report"] if config.startswith("r") else layout[split]
        if not _in_range(ids, entry).all():
            raise ValueError(f"config {config} {split}: event ids {ids.min()}-{ids.max()} outside {entry}")
    if config == "1":
        lo, hi = layout["test"]
        if not np.array_equal(np.sort(np.asarray(ids_by_split["test"], np.int64)), np.arange(lo, hi + 1)):
            raise ValueError(f"config 1 test is not exactly the event ids {layout['test']}")


def v20_config_files(file_ids, layout, config):
    """Files of each split of one config."""
    if config not in v20_config_names(layout):
        raise ValueError(f"unknown config {config}")
    by_range = {r: [] for r in V20_RANGES}
    for f, ids in file_ids.items():
        by_range[_v20_range_of(np.asarray(ids, np.int64), layout, Path(f).name)].append(f)
    all_ids = np.concatenate([np.asarray(v, np.int64) for v in file_ids.values()])
    if len(np.unique(all_ids)) != len(all_ids):
        raise ValueError("an event id appears more than once across files")
    for r in V20_RANGES:
        by_range[r].sort(key=lambda f: int(np.min(file_ids[f])))

    if config.startswith("r"):
        rep = by_range["report"]
        if len(rep) != layout["n_report"]:
            raise ValueError(f"{len(rep)} report shards, expected {layout['n_report']}")
        out = {"test": [rep[int(config[1:]) - 1]]}
    else:
        k = int(config)
        train = by_range["train"]
        if not train or len(train) % N_V20_TRAIN_CONFIGS:
            raise ValueError(f"{len(train)} train shards: not a positive multiple of {N_V20_TRAIN_CONFIGS}")
        per = len(train) // N_V20_TRAIN_CONFIGS
        out = {"train": train[(k - 1) * per : k * per]}
        if k == 1:
            out["test"] = by_range["test"]
        if k == 2:
            out["val_read"] = by_range["val_read"]
    assert_v20_splits(config, {s: np.concatenate([file_ids[f] for f in fs]) if fs else np.zeros(0, np.int64) for s, fs in out.items()}, layout)
    return out


def split_sample_v20(path, config, layout):
    """Example generators of one config's splits."""
    files = v20_config_files(v20_file_event_ids(path), layout, config)
    for s, fs in files.items():
        print(f"config {config} {s}: {[Path(f).name for f in fs]}")
    return {s: generate_examples_v20(fs) for s, fs in files.items()}


def generate_examples_v20(files: List[Path]):
    """Like generate_examples, plus event_id and gen_target, reading one event at a time."""
    for fi in files:
        pf = pq.ParquetFile(fi, pre_buffer=False, buffer_size=1 << 20)
        missing = set(V20_COLUMNS) - set(pf.schema_arrow.names)
        if missing:
            raise KeyError(f"{fi}: missing columns {sorted(missing)}")
        for batch in pf.iter_batches(batch_size=1, columns=V20_COLUMNS):
            ret = ak.from_arrow(batch)
            rec = _prepare_event(ret, 0)
            if rec is None:
                continue
            event_id = int(np.asarray(ak.to_numpy(ret["event_id"][0])).reshape(-1)[0])
            gen = np.array(_normalize_matrix(ak.to_numpy(ret["gen_target"][0]), 10), dtype=np.float32)
            gen[:, 0] = _remap_labels(gen[:, 0])
            yield (
                event_id,
                {
                    "X": rec["X"].astype(np.float32),
                    "ytarget": rec["ytarget"].astype(np.float32),
                    "ycand": rec["ycand"].astype(np.float32),
                    "genmet": rec["genmet"],
                    "genjets": rec["genjets"].astype(np.float32),
                    "targetjets": rec["targetjets"].astype(np.float32),
                    "event_id": np.int64(event_id),
                    "gen_target": gen,
                },
            )
