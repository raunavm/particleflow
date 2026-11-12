# Rewrites ColliderML v20 in the Release 1 table layout read by reader.py and postprocessing.py.
#
# v20 events are Release 1 events with the same event_id and particle_id. Per table:
#   particles     truth/particles, unchanged
#   tracks        reco/tracks, rows sorted by event_id
#   calo_hits     reco/calo_cells plus total_energy (the calibrated energy divided by the calibration
#                 factor, as postprocessing expects) and pandora_cluster (index of the Pandora cluster
#                 holding the hit, -1 if none)
#   tracker_hits  reco/tracker_hits plus particle_id, the first entry of particle_ids (-1 if empty)
#
# Output shards are contiguous event-id ranges given by --split NAME:FIRST-LAST:NSHARDS.
#
#   python3 -m mlpf.data.colliderml.v20_adapter --v20 V20/hard_scatter/ttbar/v20/parquet --out SRC \
#       --sample ttbar_pu0 --split train:0-7679:90 --split val:7680-8959:10 --split report:8960-10239:10
#   python3 -m mlpf.data.colliderml.postprocessing --input SRC --sample ttbar_pu0 --outpath CONV \
#       --shards 0:110 --algorithm pandora
#   tfds build mlpf/heptfds/colliderml_pf/ttbar_pu0_v20 --config 1 --manual_dir CONV --data_dir TFDS
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from mlpf.data.colliderml.truth import DEFAULT_CALIBRATION, calibration_factors

SOURCES = {
    "particles": ["truth/particles"],
    "tracks": ["reco/tracks"],
    "calo_hits": ["reco/calo_cells", "reco/calo_clusters"],
    "tracker_hits": ["reco/tracker_hits"],
}


def source_files(v20_parquet: Path, obj: str) -> list[tuple[int, int, Path]]:
    """(first, last, path) of each v20 file of one object, sorted by first event id."""
    out = []
    for p in (Path(v20_parquet) / obj).glob("*.parquet"):
        m = re.search(r"events(\d+)-(\d+)\.parquet$", p.name)
        out.append((int(m[1]), int(m[2]), p))
    return sorted(out)


def parse_splits(specs: list[str]) -> list[tuple[str, np.ndarray]]:
    """[(split name, event ids of one shard)] in output-shard order."""
    shards = []
    for spec in specs:
        name, rng, n = spec.split(":")
        first, last = rng.split("-")
        for ids in np.array_split(np.arange(int(first), int(last) + 1), int(n)):
            shards.append((name, ids))
    return shards


def _rows_by_event(table: pa.Table) -> dict[int, int]:
    ids = table["event_id"].to_numpy()
    if len(np.unique(ids)) != len(ids):
        raise RuntimeError("duplicate event_id in a v20 file")
    return {int(e): i for i, e in enumerate(ids)}


def _list_column(values: np.ndarray, offsets: np.ndarray, typ) -> pa.Array:
    return pa.ListArray.from_arrays(pa.array(offsets, type=pa.int32()), pa.array(values, type=typ))


def add_calo_columns(calo: pa.Table, clusters: pa.Table | None, stats: dict) -> pa.Table:
    """Append total_energy and pandora_cluster to the calo_cells rows; clusters None gives pandora_cluster -1."""
    if clusters is None:
        stats["events_without_clusters"] += calo.num_rows
    elif not np.array_equal(calo["event_id"].to_numpy(), clusters["event_id"].to_numpy()):
        raise RuntimeError("calo_cells and calo_clusters are not event-aligned")
    energy = calo["energy"].combine_chunks()
    offsets = energy.offsets.to_numpy()
    e = pc.list_flatten(energy).to_numpy(zero_copy_only=False).astype(np.float64)
    det = pc.list_flatten(calo["detector"].combine_chunks()).to_numpy(zero_copy_only=False)
    total = (e / calibration_factors(det, DEFAULT_CALIBRATION).astype(np.float64)).astype(np.float32)

    cell_id = pc.list_flatten(calo["cell_id"].combine_chunks()).to_numpy(zero_copy_only=False)
    label = np.full(len(cell_id), -1, dtype=np.int64)
    if clusters is not None:
        cl_hits = clusters["cell_ids"].combine_chunks()  # list (clusters) of list (hits)
        cl_energy = clusters["energy"].combine_chunks()
    for i in range(len(offsets) - 1 if clusters is not None else 0):
        a, b = offsets[i], offsets[i + 1]
        ids = cell_id[a:b]
        order = np.argsort(ids, kind="stable")
        stats["cell_id_duplicates"] += int((np.diff(ids[order]) == 0).sum())
        cl_list = cl_hits[i].as_py()
        n_cl = len(cl_list)
        if n_cl == 0:
            continue
        sizes = np.asarray([len(c) for c in cl_list], dtype=np.int64)
        members = np.asarray([m for c in cl_list for m in c], dtype=cell_id.dtype)
        which = np.repeat(np.arange(n_cl, dtype=np.int64), sizes)
        pos = np.searchsorted(ids[order], members)
        found = (pos < len(ids)) & (ids[order][np.minimum(pos, len(ids) - 1)] == members)
        stats["cluster_hits_missing"] += int((~found).sum())
        rows = order[pos[found]]
        which = which[found]
        # a hit listed by two clusters goes to the more energetic one
        ecl = np.asarray(cl_energy[i].as_py(), dtype=np.float64)
        srt = np.lexsort((-ecl[which], rows))
        rows, which = rows[srt], which[srt]
        first = np.ones(len(rows), dtype=bool)
        first[1:] = rows[1:] != rows[:-1]
        stats["hits_in_two_clusters"] += int((~first).sum())
        label[a + rows[first]] = which[first]
        stats["clusters"] += n_cl
    stats["hits"] += len(cell_id)
    stats["hits_unclustered"] += int((label < 0).sum())
    calo = calo.append_column("total_energy", _list_column(total, offsets, pa.float32()))
    return calo.append_column("pandora_cluster", _list_column(label, offsets, pa.int64()))


def add_tracker_particle_id(hits: pa.Table) -> pa.Table:
    lists = hits["particle_ids"].combine_chunks()  # list (hits) of list (particles)
    out_offsets = lists.offsets.to_numpy()
    inner = lists.values
    first = np.full(len(inner), -1, dtype=np.int64)
    n = pc.list_value_length(inner).to_numpy(zero_copy_only=False)
    starts = inner.offsets.to_numpy()[:-1]
    vals = pc.list_flatten(inner).to_numpy(zero_copy_only=False)
    has = n > 0
    first[has] = vals[starts[has]].astype(np.int64)
    return hits.append_column("particle_id", _list_column(first, out_offsets, pa.int64()))


def convert(v20_parquet: Path, out: Path, sample: str, splits: list[str], no_pandora_clusters: bool = False) -> dict:
    """Write the four tables as shards of the given splits and return a summary."""
    shards = parse_splits(splits)
    n_shards = len(shards)
    wanted = np.concatenate([ids for _, ids in shards])
    if len(np.unique(wanted)) != len(wanted):
        raise ValueError("splits overlap")
    shard_of = {int(e): k for k, (_, ids) in enumerate(shards) for e in ids}
    report = {"sample": sample, "splits": splits, "shards": [{"split": n, "first": int(i[0]), "last": int(i[-1]), "n": len(i)} for n, i in shards]}

    for obj, srcs in SOURCES.items():
        stats = {
            "hits": 0,
            "hits_unclustered": 0,
            "clusters": 0,
            "hits_in_two_clusters": 0,
            "cluster_hits_missing": 0,
            "cell_id_duplicates": 0,
            "events_without_clusters": 0,
        }
        odir = Path(out) / f"{sample}_{obj}" / "data" / f"{sample}_{obj}"
        odir.mkdir(parents=True, exist_ok=True)
        buf: dict[int, list] = {}
        done = 0
        files = source_files(v20_parquet, srcs[0])
        extra = {f[0]: f[2] for f in source_files(v20_parquet, srcs[1])} if len(srcs) > 1 else {}
        for first_ev, last_ev, path in files:
            need = [e for e in wanted if first_ev <= e <= last_ev]
            if not need:
                continue
            table = pq.read_table(path)
            rows = _rows_by_event(table)
            missing = [e for e in need if e not in rows]
            if missing:
                raise RuntimeError(f"{path.name}: events {missing[:5]} missing")
            take = pa.array([rows[e] for e in need], type=pa.int64())
            part = table.take(take)
            del table
            if obj == "calo_hits" and first_ev not in extra:
                if not no_pandora_clusters:
                    raise RuntimeError(f"{path.name}: no reco/calo_clusters file, see --no-pandora-clusters")
                part = add_calo_columns(part, None, stats)
            elif obj == "calo_hits":
                cl = pq.read_table(extra[first_ev])
                cl_rows = _rows_by_event(cl)
                part = add_calo_columns(part, cl.take(pa.array([cl_rows[e] for e in need], type=pa.int64())), stats)
                del cl
            elif obj == "tracker_hits":
                part = add_tracker_particle_id(part)
            ids = part["event_id"].to_numpy()
            for k in sorted({shard_of[int(e)] for e in ids}):
                sel = np.flatnonzero([shard_of[int(e)] == k for e in ids])
                buf.setdefault(k, []).append(part.take(pa.array(sel, type=pa.int64())))
                got = sum(t.num_rows for t in buf[k])
                if got == len(shards[k][1]):
                    t = pa.concat_tables(buf.pop(k)).combine_chunks()
                    t = t.take(pa.array(np.argsort(t["event_id"].to_numpy(), kind="stable"), type=pa.int64()))
                    if not np.array_equal(t["event_id"].to_numpy(), shards[k][1]):
                        raise RuntimeError(f"shard {k} of {obj}: event ids differ from the plan")
                    pq.write_table(t, odir / f"train-{k:05d}-of-{n_shards:05d}.parquet")
                    done += 1
            del part
        if buf or done != n_shards:
            raise RuntimeError(f"{obj}: wrote {done} of {n_shards} shards")
        if obj == "calo_hits":
            report["calo"] = stats
        print(f"{obj}: {done} shards", flush=True)
    (Path(out) / "v20_adapter_report.json").write_text(json.dumps(report, indent=1))
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v20", required=True, help="the v20 parquet/ directory, e.g. .../hard_scatter/ttbar/v20/parquet")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sample", default="ttbar_pu0")
    ap.add_argument("--split", action="append", required=True, help="NAME:FIRST-LAST:NSHARDS (event ids inclusive); repeatable")
    ap.add_argument("--no-pandora-clusters", action="store_true", help="allow files without reco/calo_clusters (pandora_cluster -1)")
    a = ap.parse_args()
    rep = convert(Path(a.v20), Path(a.out), a.sample, a.split, a.no_pandora_clusters)
    print(json.dumps(rep.get("calo", {})))


if __name__ == "__main__":
    main()
