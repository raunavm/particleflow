"""Score the ColliderML v20 Pandora PFOs, and optionally MLPF predictions, against the converter's gen_target.

Candidates are matched to targets with upstream's validation matcher (mlpf/model/validation_metrics.py).
MLPF cuts on the presence logit are placed on --cut-events at 0.5, 1 and 2 times Pandora's neutral fake rate
and read on --read-events. Intervals are 68% event-bootstrap intervals. MLPF predictions carry no event id and
are paired to events by gen MET; events with a shared gen MET are dropped.

Usage on v20 PU0 converter output, with the cut and read events of the TFDS configs 1 and 2:

    python3 scripts/colliderml_v20/score.py --targets CONV --pandora V20/reco/pfos --cut-events 7680-8319
        --read-events 8320-8959 [--predictions NAME=PRED_DIR] --out pandora.json
"""

import argparse
import glob
import json
import math
from pathlib import Path

import awkward as ak
import fastjet
import numpy as np
import pyarrow.parquet as pq
import torch
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from mlpf.data.target_building import map_pdgid_to_candid
from mlpf.model.validation_metrics import MATCH_COST_CAP, MATCH_DR, MATCH_LOG_PT, MATCH_REL_PT, _accumulate_event_metrics, _empty_metrics

LABELS = [0, 211, 130, 22, 11, 13]
NEUTRAL = (2, 3)
CHARGED = (1, 4, 5)
NUM_CLASSES = len(LABELS)
TARGET_FID = (1.0, 2.4)
CAND_FID = (0.5, 2.6)
ANCHORS = (0.5, 1.0, 2.0)
JET_DEF = fastjet.JetDefinition(fastjet.antikt_algorithm, 0.4)


def match(t, c):
    """Upstream's capped assignment, solved per connected component. Returns (cand index, target index) of accepted pairs."""
    nt, nc = len(t["pt"]), len(c["pt"])
    if nt == 0 or nc == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    deta = c["eta"][:, None] - t["eta"][None, :]
    dphi = np.remainder(c["phi"][:, None] - t["phi"][None, :] + math.pi, 2 * math.pi) - math.pi
    dr = np.sqrt(deta**2 + dphi**2)
    lpt = np.abs(np.log(np.maximum(c["pt"], 1e-8))[:, None] - np.log(np.maximum(t["pt"], 1e-8))[None, :])
    cost = np.minimum((dr / MATCH_DR) ** 2 + (lpt / MATCH_LOG_PT) ** 2, MATCH_COST_CAP)
    ci, ti = np.nonzero(cost < MATCH_COST_CAP)
    if len(ci) == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    g = coo_matrix((np.ones(len(ci)), (ci, nc + ti)), shape=(nc + nt, nc + nt))
    _, comp = connected_components(g, directed=False)
    out_c, out_t = [], []
    for k in np.unique(comp[ci]):
        cs = np.flatnonzero(comp[:nc] == k)
        ts = np.flatnonzero(comp[nc:] == k)
        r, s = linear_sum_assignment(cost[np.ix_(cs, ts)])
        out_c.append(cs[r])
        out_t.append(ts[s])
    mc, mt = np.concatenate(out_c), np.concatenate(out_t)
    rel = np.abs(c["pt"][mc] - t["pt"][mt]) / np.maximum(t["pt"][mt], 1e-8)
    ok = (dr[mc, mt] < MATCH_DR) & (rel < MATCH_REL_PT)
    return mc[ok], mt[ok]


def load_targets(conv_dir, events):
    """gen_target, genmet and genjet per event_id from the converter output."""
    out = {}
    for f in sorted(glob.glob(f"{conv_dir}/*.parquet")):
        t = pq.read_table(f, columns=["event_id", "genmet", "gen_target", "genjet"])
        for i, eid in enumerate(t["event_id"].to_pylist()):
            if events is None or eid in events:
                out[eid] = {
                    "gen": np.asarray(t["gen_target"][i].as_py(), np.float64).reshape(-1, 10),
                    "genmet": np.float32(t["genmet"][i].as_py()),
                    "genjet": np.asarray(t["genjet"][i].as_py(), np.float64).reshape(-1, 4),
                }
    return out


def targets_of(ev):
    g = ev["gen"]
    cls = np.array([LABELS.index(int(v)) if int(v) in LABELS else 0 for v in g[:, 0]], np.int64)
    return {"cls": cls, "pt": g[:, 2], "eta": g[:, 3], "phi": np.arctan2(g[:, 4], g[:, 5]), "energy": g[:, 6]}


def load_pandora(pfo_dir, events):
    """Candidates per event_id from the v20 pfos table."""
    out = {}
    for f in sorted(glob.glob(f"{pfo_dir}/*.parquet")):
        t = pq.read_table(f, columns=["event_id", "pdg", "charge", "energy", "px", "py", "pz"])
        for i, eid in enumerate(t["event_id"].to_pylist()):
            if eid not in events:
                continue
            px, py, pz, e = (np.asarray(t[k][i].as_py(), np.float64) for k in ("px", "py", "pz", "energy"))
            pdg, q = np.asarray(t["pdg"][i].as_py()), np.asarray(t["charge"][i].as_py(), np.float64)
            pt = np.hypot(px, py)
            cls = np.array([LABELS.index(map_pdgid_to_candid(abs(int(p)), c)) for p, c in zip(pdg, q)], np.int64)
            out[eid] = {
                "cls": cls,
                "pt": pt,
                "eta": np.arcsinh(pz / np.maximum(pt, 1e-12)),
                "phi": np.arctan2(py, px),
                "energy": e,
                "d": np.ones(len(pt)),
                "p": np.ones(len(pt)),
            }
    return out


def load_mlpf(pred_dir, genmet_to_event):
    """Candidates per event_id from upstream test output."""
    out = {}
    for f in sorted(glob.glob(f"{pred_dir}/**/pred_*.parquet", recursive=True)):
        a = ak.from_parquet(f)
        for i in range(len(a)):
            key = np.float32(np.asarray(a["genmet"][i]).reshape(-1)[0])
            if key not in genmet_to_event:
                continue
            e = genmet_to_event[key]
            if e in out:
                raise ValueError(f"event {e} predicted twice under {pred_dir}")
            p = a["particles"]["pred"][i]
            cb = np.asarray(ak.to_numpy(p["cls_binary"]), np.float64)
            oh = np.asarray(ak.to_numpy(p["cls_id_onehot"]), np.float64)
            p4 = np.asarray(ak.to_numpy(p["p4"]), np.float64)
            d = cb[:, 1] - cb[:, 0]
            out[e] = {
                "d": d,
                "p": 1.0 / (1.0 + np.exp(-d)),
                "cls": np.argmax(oh[:, 1:], axis=1) + 1,
                "pt": p4[:, 0],
                "eta": p4[:, 1],
                "phi": p4[:, 2],
                "energy": p4[:, 3],
            }
    return out


REACH = MATCH_DR * math.sqrt(MATCH_COST_CAP)  # pairs further apart always cost the cap


def prepare(t, c):
    """Per-event cache of the candidates that can be matched and the fiducial masks."""
    tfid = (t["pt"] > TARGET_FID[0]) & (np.abs(t["eta"]) < TARGET_FID[1])
    near = np.zeros(len(c["pt"]), bool)
    if len(t["pt"]) and len(c["pt"]):
        pts = np.concatenate([np.stack([t["eta"], t["phi"] + s], 1) for s in (-2 * math.pi, 0.0, 2 * math.pi)])
        near = cKDTree(pts).query(np.stack([c["eta"], c["phi"]], 1), k=1, distance_upper_bound=REACH)[0] < REACH
    return {
        "t": t,
        "c": c,
        "near": near,
        "tneu": tfid & np.isin(t["cls"], NEUTRAL),
        "tch": tfid & np.isin(t["cls"], CHARGED),
        "cneu_fid": np.isin(c["cls"], NEUTRAL) & (c["pt"] > CAND_FID[0]) & (np.abs(c["eta"]) < CAND_FID[1]),
    }


def event_counts(ev, cut):
    """[found neutral, n neutral, fakes neutral, found charged, n charged] at candidates d > cut."""
    t, c = ev["t"], ev["c"]
    sel = c["d"] > cut
    idx = np.flatnonzero(sel & ev["near"])
    mc, mt = match(t, {k: c[k][idx] for k in ("pt", "eta", "phi")})
    mc = idx[mc]
    found_neu = int((ev["tneu"][mt] & np.isin(c["cls"][mc], NEUTRAL)).sum())
    found_ch = int((ev["tch"][mt] & np.isin(c["cls"][mc], CHARGED)).sum())
    matched = np.zeros(len(c["pt"]), bool)
    matched[mc] = True
    fakes = int((sel & ev["cneu_fid"] & ~matched).sum())
    return np.array([found_neu, ev["tneu"].sum(), fakes, found_ch, ev["tch"].sum()], np.int64)


def totals(counts):
    s = counts.sum(axis=0)
    return {"eff_neutral": s[0] / max(s[1], 1), "fakes_neutral_per_event": s[2] / len(counts), "eff_charged": s[3] / max(s[4], 1)}


def place_cut(prepared, events, rate, lo=-20.0, hi=20.0, iters=30):
    """Bisect the cut so the neutral fake rate on events equals rate."""
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        f = np.mean([event_counts(prepared[e], mid)[2] for e in events])
        lo, hi = (mid, hi) if f > rate else (lo, mid)
    return 0.5 * (lo + hi)


def bootstrap(per_event_a, per_event_b=None, n=1000, seed=0):
    """68% interval of eff_neutral, or of a - b paired by event."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(per_event_a), len(per_event_a))
        a = per_event_a[idx].sum(axis=0)
        v = a[0] / max(a[1], 1)
        if per_event_b is not None:
            b = per_event_b[idx].sum(axis=0)
            v -= b[0] / max(b[1], 1)
        vals.append(v)
    return [float(np.percentile(vals, 16)), float(np.percentile(vals, 84))]


def grid_counts(prepared, events, grid):
    """event_counts on a grid of cuts."""
    return np.array([[event_counts(prepared[e], g) for g in grid] for e in events])


def paired_eff_ci(counts, ref_counts, num, den, n=1000, seed=0):
    """68% interval of the paired efficiency difference to ref_counts."""
    rng = np.random.default_rng(seed)

    def eff(c, i):
        return c[i, num].sum() / max(c[i, den].sum(), 1)

    d = [eff(counts, i) - eff(ref_counts, i) for i in (rng.integers(0, len(counts), len(counts)) for _ in range(n))]
    return np.percentile(d, [16, 84]).tolist()


def matched_bootstrap(cut_counts, read_counts, grid, factor, ref_cut_fakes, ref_read=None, n=1000, seed=0):
    """68% interval of eff_neutral at factor times the reference fake rate, re-placing the cut in every resample."""
    rng = np.random.default_rng(seed)
    vals, clamped = [], 0
    for _ in range(n):
        ic = rng.integers(0, len(cut_counts), len(cut_counts))
        rate = factor * ref_cut_fakes[ic].mean()
        f = np.minimum.accumulate(cut_counts[ic, :, 2].mean(axis=0))
        clamped += not (f[-1] <= rate <= f[0])
        cut = np.interp(rate, f[::-1], grid[::-1])
        ir = rng.integers(0, len(read_counts), len(read_counts))
        s = read_counts[ir].sum(axis=0)
        v = np.interp(cut, grid, s[:, 0] / np.maximum(s[:, 1], 1))
        if ref_read is not None:
            r = ref_read[ir].sum(axis=0)
            v -= r[0] / max(r[1], 1)
        vals.append(v)
    return [float(np.percentile(vals, 16)), float(np.percentile(vals, 84))], clamped / n


def upstream_metrics(targets, cands, events, cut):
    """Upstream's validation metrics."""
    m = _empty_metrics(NUM_CLASSES)

    def conv(d, s):
        return {k: torch.as_tensor(d[k][s], dtype=torch.float32) for k in ("pt", "eta", "energy")} | {
            "sin_phi": torch.as_tensor(np.sin(d["phi"][s]), dtype=torch.float32),
            "cos_phi": torch.as_tensor(np.cos(d["phi"][s]), dtype=torch.float32),
            "phi": torch.as_tensor(d["phi"][s], dtype=torch.float32),
            "cls_id": torch.as_tensor(d["cls"][s], dtype=torch.long),
        }

    for e in events:
        t = targets_of(targets[e])
        c = cands[e]
        _accumulate_event_metrics(m, conv(t, np.ones(len(t["pt"]), bool)), conv(c, c["d"] > cut), NUM_CLASSES)
    keep = [
        "matching/efficiency",
        "matching/purity",
        "matching/f1",
        "matching/duplicate_fraction",
        "matched/pid_accuracy",
        "matched/pt_relative_abs_mean",
        "event/energy_response_mean",
        "event/met_abs_error",
    ] + [f"class_{k}/efficiency" for k in range(1, NUM_CLASSES)]
    return {k: (m[k][0] / m[k][1] if m[k][1] else float("nan")) for k in keep}


def jet_response(targets, cands, events, cut, min_pt=30.0, max_eta=2.5, dr=0.2, weighted=False, per_event=False):
    """Jet response of the candidates with d > cut, or with weighted of all candidates scaled by p."""
    resp, share, per = [], [], []
    for e in events:
        per.append(len(resp))
        c = cands[e]
        if weighted:
            with np.errstate(over="ignore", invalid="ignore"):
                pe, pz = c["p"] * c["energy"], c["p"] * c["pt"] * np.sinh(c["eta"])
            sel = (c["p"] > 0) & np.isfinite(c["pt"]) & np.isfinite(c["phi"]) & np.isfinite(pz) & np.isfinite(pe)
            w = c["p"][sel]
            if sel.any() and pe[sel].sum() > 0:
                share.append(float(pe[sel].max() / pe[sel].sum()))
        else:
            sel = c["d"] > cut
            w = 1.0
        if not sel.any():
            continue
        pt = c["pt"][sel] * w
        vec = ak.zip(
            {"px": pt * np.cos(c["phi"][sel]), "py": pt * np.sin(c["phi"][sel]), "pz": pt * np.sinh(c["eta"][sel]), "E": c["energy"][sel] * w}
        )
        jets = fastjet.ClusterSequence(vec, JET_DEF).inclusive_jets(min_pt=3.0)
        jpx, jpy, jpz = (np.asarray(ak.to_numpy(jets[k]), np.float64) for k in ("px", "py", "pz"))
        jpt = np.hypot(jpx, jpy)
        jeta, jphi = np.arcsinh(jpz / np.maximum(jpt, 1e-9)), np.arctan2(jpy, jpx)
        for g in targets[e]["genjet"]:
            if g[0] < min_pt or abs(g[1]) > max_eta or len(jpt) == 0:
                continue
            d = np.hypot(jeta - g[1], np.remainder(jphi - g[2] + math.pi, 2 * math.pi) - math.pi)
            k = int(np.argmin(d))
            if d[k] < dr:
                resp.append(jpt[k] / g[0])
    if per_event:
        bounds = per + [len(resp)]
        return [np.asarray(resp[bounds[i] : bounds[i + 1]], np.float64) for i in range(len(events))]
    r = np.asarray(resp)
    extra = {"max_candidate_pE_share_p50_p99_max": [float(x) for x in np.percentile(share, [50, 99, 100])]} if weighted and share else {}
    if len(r) == 0:
        return {"jets_matched": 0} | extra
    q16, q50, q84 = np.percentile(r, [16, 50, 84])
    return {"jets_matched": int(len(r)), "jet_response_median": float(q50), "jet_resolution_iqr68_over_median": float((q84 - q16) / 2 / q50)} | extra


def jet_summary(per, ref=None, n=1000, seed=0):
    """Jet response and resolution, with intervals on the resolution and on its difference to ref."""

    def res(p, idx):
        r = np.concatenate([p[i] for i in idx] + [np.zeros(0)])
        if len(r) == 0:
            return float("nan")
        q16, q50, q84 = np.percentile(r, [16, 50, 84])
        return (q84 - q16) / 2 / q50

    a = np.concatenate(per + [np.zeros(0)])
    if len(a) == 0:
        return {"jets_matched": 0}
    rng = np.random.default_rng(seed)
    draws = [rng.integers(0, len(per), len(per)) for _ in range(n)]
    boot = np.array([res(per, d) for d in draws])
    out = {
        "jets_matched": int(len(a)),
        "jet_response_median": float(np.median(a)),
        "jet_resolution_iqr68_over_median": res(per, range(len(per))),
        "jet_resolution_ci68": np.nanpercentile(boot, [16, 84]).tolist(),
    }
    if ref is not None:
        out["minus_pandora_ci68"] = np.nanpercentile(boot - np.array([res(ref, d) for d in draws]), [16, 84]).tolist()
    return out


def paired_jets(targets, cands, read_ev, cut, p_jets):
    """Jet resolution minus Pandora's, at the 1x cut and weighted by presence."""
    return {
        "selected_at_1x_pandora": jet_summary(jet_response(targets, cands, read_ev, cut, per_event=True), ref=p_jets),
        "presence_weighted": jet_summary(jet_response(targets, cands, read_ev, 0.0, weighted=True, per_event=True), ref=p_jets),
        "pandora": jet_summary(p_jets),
    }


def score_predictions(targets, cands, cut_ev, read_ev, f_p, p_cut, p_read, p_jets):
    """All metrics of one MLPF prediction set."""
    out = {
        "presence_weighted": jet_response(targets, cands, read_ev, 0.0, weighted=True),
        "argmax": {"upstream_metrics": upstream_metrics(targets, cands, read_ev, 0.0), **jet_response(targets, cands, read_ev, 0.0)},
    }
    prep = {e: prepare(targets_of(targets[e]), cands[e]) for e in cut_ev + read_ev}
    lo_b, hi_b = -20.0, 20.0
    cuts = {x: place_cut(prep, cut_ev, x * f_p, lo_b, hi_b) for x in ANCHORS}
    reached = {x: bool(lo_b + 1e-3 < c < hi_b - 1e-3) for x, c in cuts.items()}
    span = [c for x, c in cuts.items() if reached[x]] or list(cuts.values())
    pad = max(0.5 * (max(span) - min(span)), 0.5)
    grid = np.linspace(min(span) - pad, max(span) + pad, 49)
    g_cut, g_read = grid_counts(prep, cut_ev, grid), grid_counts(prep, read_ev, grid)
    s_read = g_read.sum(axis=0)
    for x, cut in cuts.items():
        per = np.array([event_counts(prep[e], cut) for e in read_ev])
        ci, clamped = matched_bootstrap(g_cut, g_read, grid, x, p_cut[:, 2])
        gi = float(np.interp(cut, grid, s_read[:, 0] / np.maximum(s_read[:, 1], 1)))
        if not reached[x]:
            ci, gi = None, None
        out[f"at_{x}xPandora"] = {
            "cut": cut,
            "rate_reached": reached[x],
            "target_fake_rate": x * f_p,
            "fakes_on_cut_events": totals(np.array([event_counts(prep[e], cut) for e in cut_ev]))["fakes_neutral_per_event"],
            "read": totals(per),
            "eff_neutral_grid_interp": gi,
            "eff_neutral_ci68": ci,
            "bootstrap_cut_outside_grid": clamped,
            "minus_pandora_ci68": matched_bootstrap(g_cut, g_read, grid, x, p_cut[:, 2], p_read)[0] if x == 1.0 and reached[x] else None,
            **jet_response(targets, cands, read_ev, cut),
        }
        if x == 1.0:
            out[f"at_{x}xPandora"]["eff_charged_minus_pandora_ci68"] = paired_eff_ci(per, p_read, 3, 4)
    out["jets_paired_vs_pandora"] = paired_jets(targets, cands, read_ev, cuts[1.0], p_jets)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", required=True, help="converter output directory covering the cut and read events")
    ap.add_argument("--pandora", required=True, help="v20 reco/pfos directory")
    ap.add_argument("--predictions", action="append", default=[], help="NAME=PRED_DIR (upstream MLPF test output); repeatable")
    ap.add_argument("--cut-events", required=True, help="FIRST-LAST event ids where cuts are placed")
    ap.add_argument("--read-events", required=True, help="FIRST-LAST event ids where results are read")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    def rng(s):
        return set(range(int(s.split("-")[0]), int(s.split("-")[1]) + 1))

    cut_ev, read_ev = rng(a.cut_events), rng(a.read_events)
    if cut_ev & read_ev:
        raise SystemExit("cut and read events overlap")
    targets = load_targets(a.targets, cut_ev | read_ev)
    if set(targets) != cut_ev | read_ev:
        raise SystemExit(f"targets missing for {len((cut_ev | read_ev) - set(targets))} events")
    by_gm = {}
    for e, v in targets.items():
        by_gm.setdefault(v["genmet"], []).append(e)
    shared = sorted(e for es in by_gm.values() if len(es) > 1 for e in es)
    gm = {k: es[0] for k, es in by_gm.items() if len(es) == 1}
    cut_ev, read_ev = sorted(cut_ev - set(shared)), sorted(read_ev - set(shared))

    res = {"definition": __doc__, "n_cut_events": len(cut_ev), "n_read_events": len(read_ev), "dropped_events_shared_genmet": shared, "results": {}}
    pandora = load_pandora(a.pandora, set(targets))
    prep = {e: prepare(targets_of(targets[e]), pandora[e]) for e in targets}
    p_cut = np.array([event_counts(prep[e], 0.0) for e in cut_ev])
    p_read = np.array([event_counts(prep[e], 0.0) for e in read_ev])
    f_p = totals(p_cut)["fakes_neutral_per_event"]
    res["pandora_fake_rate_on_cut_events"] = f_p
    res["results"]["pandora"] = {
        "read": totals(p_read),
        "eff_neutral_ci68": bootstrap(p_read),
        "upstream_metrics": upstream_metrics(targets, pandora, read_ev, 0.0),
        **jet_response(targets, pandora, read_ev, 0.0),
        "presence_weighted": jet_response(targets, pandora, read_ev, 0.0, weighted=True),
    }
    specs = [s.split("=", 1) for s in a.predictions]
    names = [n for n, _ in specs]
    if "pandora" in names or len(set(names)) != len(names):
        raise SystemExit(f"prediction names must be unique and not 'pandora': {names}")
    p_jets = jet_response(targets, pandora, read_ev, 0.0, per_event=True) if specs else None
    for name, d in specs:
        cands = load_mlpf(d, gm)
        missing = (set(cut_ev) | set(read_ev)) - set(cands)
        if missing:
            raise SystemExit(f"{name}: predictions missing for {len(missing)} events")
        res["results"][name] = score_predictions(targets, cands, cut_ev, read_ev, f_p, p_cut, p_read, p_jets)
    Path(a.out).write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: v.get("read", v.get("at_1.0xPandora", {}).get("read")) for k, v in res["results"].items()}, indent=1, default=float))


if __name__ == "__main__":
    main()
