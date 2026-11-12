import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from scipy.optimize import linear_sum_assignment

from mlpf.model.validation_metrics import _matching_cost, _pairwise_geometry

_spec = importlib.util.spec_from_file_location("colliderml_v20_score", Path(__file__).parents[1] / "scripts" / "colliderml_v20" / "score.py")
score = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(score)


def _ev(rows):
    return {
        k: np.array([r[i] for r in rows], dtype=np.int64 if k == "cls" else np.float64) for i, k in enumerate(("cls", "pt", "eta", "phi", "energy"))
    }


def test_match_equals_upstream_dense_assignment():
    rng = np.random.default_rng(3)
    for trial in range(20):
        nt, nc = rng.integers(1, 60), rng.integers(1, 90)
        t = {"pt": rng.uniform(0.5, 30, nt), "eta": rng.uniform(-1, 1, nt), "phi": rng.uniform(-math.pi, math.pi, nt)}
        src = rng.integers(0, nt, nc)
        c = {
            "pt": t["pt"][src] * rng.uniform(0.4, 1.8, nc),
            "eta": t["eta"][src] + rng.normal(0, 0.06, nc),
            "phi": np.remainder(t["phi"][src] + rng.normal(0, 0.06, nc) + math.pi, 2 * math.pi) - math.pi,
        }
        t32 = {k: v.astype(np.float32).astype(np.float64) for k, v in t.items()}
        c32 = {k: v.astype(np.float32).astype(np.float64) for k, v in c.items()}
        mc, mt = score.match(t32, c32)
        tt = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in t.items()}
        cc = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in c.items()}
        _, _, dr, lpt, rel = _pairwise_geometry(tt, cc)
        pi, ti = linear_sum_assignment(_matching_cost(dr, lpt).numpy())
        ok = ((dr[pi, ti] < 0.1) & (rel[pi, ti] < 0.5)).numpy()
        assert set(zip(mc.tolist(), mt.tolist())) == set(zip(pi[ok].tolist(), ti[ok].tolist())), trial


def test_event_counts():
    # photon, neutral hadron and charged hadron in the fiducial region, and a photon outside it
    t = _ev([(3, 10, 0.0, 0.0, 10), (2, 5, 1.0, 1.0, 5), (1, 8, -1.0, -2.0, 8), (3, 4, 2.45, 0.5, 4)])
    c = _ev(
        [
            (3, 10.5, 0.01, 0.0, 10.5),  # found
            (2, 3.0, -2.0, 2.5, 3.0),  # fake
            (2, 8.2, -1.0, -2.01, 8.2),  # matched to the charged hadron, neither found nor fake
            (1, 2.0, 0.5, -3.0, 2.0),  # charged, not a neutral fake
            (2, 0.3, 0.3, 0.3, 0.3),  # below the candidate pt cut
        ]
    )
    c["d"] = np.full(5, 5.0)
    ev = score.prepare(t, c)
    assert score.event_counts(ev, 0.0).tolist() == [1, 2, 1, 0, 1]
    assert score.event_counts(ev, 9.0).tolist() == [0, 2, 0, 0, 1]


def test_place_cut():
    rng = np.random.default_rng(5)
    prep = {}
    for e in range(30):
        n = 40
        c = {"cls": np.full(n, 2), "pt": rng.uniform(1, 5, n), "eta": rng.uniform(1, 5, n), "phi": rng.uniform(-3, 3, n), "energy": np.ones(n)}
        c["d"] = rng.normal(0, 3, n)
        prep[e] = score.prepare(_ev([(3, 10, 0.0, 0.0, 10)]), c)
    cut = score.place_cut(prep, list(prep), 2.0)
    assert abs(np.mean([score.event_counts(prep[e], cut)[2] for e in prep]) - 2.0) < 0.1


def test_presence_weighted_equals_cut_when_p_is_one():
    rng = np.random.default_rng(7)
    targets, cands = {}, {}
    for e in range(20):
        n = 40
        pt = rng.exponential(8, n) + 0.5
        eta, phi = rng.uniform(-2, 2, n), rng.uniform(-math.pi, math.pi, n)
        hard = np.argsort(pt)[-2:]
        pt[hard] += 40
        targets[e] = {"genjet": np.stack([pt[hard], eta[hard], phi[hard], pt[hard] * np.cosh(eta[hard])], 1)}
        cands[e] = {"pt": pt, "eta": eta, "phi": phi, "energy": pt * np.cosh(eta), "d": np.ones(n), "p": np.ones(n), "cls": np.ones(n, int)}
    flow = score.jet_response(targets, cands, range(20), 0.0, weighted=True)
    flow.pop("max_candidate_pE_share_p50_p99_max")
    assert flow == score.jet_response(targets, cands, range(20), 0.0)
    assert score.jet_summary(score.jet_response(targets, cands, range(20), 0.0, per_event=True), ref=None)["jets_matched"] == flow["jets_matched"] > 0


def test_main_pandora_equal_to_targets(tmp_path, monkeypatch):
    # PFOs that are the targets themselves: every fiducial target found, no fakes
    gen, pfos = [], []
    for e in range(6):
        g = np.array(
            [[22, 0, 10, 0.1, 0, 1, 10, 0, 0, 1], [211, 1, 20, -0.5, 1, 0, 23, 0, 1, 0], [130, 0, 5 + e, 1.0, 0, -1, 8, 0, 0, 1]], np.float32
        )
        gen.append(g.tolist())
        pt, eta, phi = g[:, 2], g[:, 3], np.arctan2(g[:, 4], g[:, 5])
        pfos.append(
            {
                "pdg": [22, 211, 2112],
                "charge": [0.0, 1.0, 0.0],
                "energy": g[:, 6].tolist(),
                "px": (pt * np.cos(phi)).tolist(),
                "py": (pt * np.sin(phi)).tolist(),
                "pz": (pt * np.sinh(eta)).tolist(),
            }
        )
    (tmp_path / "conv").mkdir()
    (tmp_path / "pfos").mkdir()
    pq.write_table(
        pa.table({"event_id": list(range(6)), "genmet": [float(e) for e in range(6)], "gen_target": gen, "genjet": [[]] * 6}),
        tmp_path / "conv" / "a.parquet",
    )
    pq.write_table(pa.table({"event_id": list(range(6)), **{k: [p[k] for p in pfos] for k in pfos[0]}}), tmp_path / "pfos" / "a.parquet")
    out = tmp_path / "score.json"
    argv = [
        "score.py",
        "--targets",
        str(tmp_path / "conv"),
        "--pandora",
        str(tmp_path / "pfos"),
        "--cut-events",
        "0-2",
        "--read-events",
        "3-5",
        "--out",
        str(out),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    score.main()
    read = json.loads(out.read_text())["results"]["pandora"]["read"]
    assert read == {"eff_neutral": 1.0, "fakes_neutral_per_event": 0.0, "eff_charged": 1.0}
