"""(eta, phi) crops of large events for set-output training.

An event is tiled into len(eta_edges) - 1 eta bins (the outer two open-ended, so every object has one interior) times
nphi phi bins from -pi. Crop c = eta_bin * nphi + phi_bin. A crop keeps the elements inside its interior widened by a
guard band (guard elements are input only) and the targets inside its interior.

Calorimeter elements sit at their own (eta, phi). Tracks sit at their ECAL impact point seen from the origin, or at their
own direction when they do not reach the ECAL. A charged target with a track (gen_target_track >= 0) takes that track's
position, an unlinked charged target the impact point of its gen helix, and a neutral target its own direction.
"""

import math

import numpy as np

from mlpf.conf import EDM4HEP

ETA_EDGES = (-np.inf, -2.4, -0.8, 0.8, 2.4, np.inf)
NPHI = 4
TWO_PI = 2 * math.pi

# OpenDataDetector tracker and ECAL front faces [mm]
TRACKER_R = 1080.0
TRACKER_Z = 3030.0
B_INNER = EDM4HEP.DETECTORS["colliderml"].b_field
ECAL_BARREL_R = 1252.0
ECAL_ENDCAP_Z = 3202.0

# gen_target columns (postprocessing.GEN_TARGET_ORDER)
GEN_CHARGE, GEN_PT, GEN_ETA, GEN_SIN_PHI, GEN_COS_PHI = 1, 2, 3, 4, 5

PER_ELEMENT_KEYS = ("X", "ytarget", "ycand")
PER_TARGET_KEYS = ("ytarget_set", "gen_target")


def n_crops(eta_edges=ETA_EDGES, nphi=NPHI):
    return (len(eta_edges) - 1) * nphi


def extrapolate_to_ecal(d0, z0, phi, theta, qop):
    """Propagate perigee track parameters to the ECAL front face: helix to the tracker exit, then a straight line.

    Returns (x, y, z, reaches_ecal) in mm. Tracks that curl up inside the tracker have reaches_ecal False and position 0.
    """
    d0 = np.asarray(d0, dtype=np.float64)
    z0 = np.asarray(z0, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    qop = np.asarray(qop, dtype=np.float64)

    charge = np.where(qop >= 0.0, 1.0, -1.0)
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    pt = sin_t / np.abs(qop)
    radius = pt / (0.3 * B_INNER) * 1000.0

    # perigee point (ACTS: at d0 along (-sin phi, cos phi, 0)) and helix center (B along +z, positive charge turns
    # clockwise); the angle center->particle advances as alpha(s) = alpha0 - charge * s / R
    x0 = -d0 * np.sin(phi)
    y0 = d0 * np.cos(phi)
    cx = x0 + radius * charge * np.sin(phi)
    cy = y0 - radius * charge * np.cos(phi)
    alpha0 = np.arctan2(y0 - cy, x0 - cx)

    # tracker barrel r = TRACKER_R
    d_center = np.hypot(cx, cy)
    with np.errstate(divide="ignore", invalid="ignore"):
        cos_beta = (radius**2 + d_center**2 - TRACKER_R**2) / (2.0 * radius * d_center)
    crosses_barrel = np.isfinite(cos_beta) & (np.abs(cos_beta) <= 1.0)
    beta = np.arccos(np.clip(cos_beta, -1.0, 1.0))
    alpha_to_origin = np.arctan2(-cy, -cx)
    s_a = radius * np.mod(charge * (alpha0 - (alpha_to_origin + beta)), TWO_PI)
    s_b = radius * np.mod(charge * (alpha0 - (alpha_to_origin - beta)), TWO_PI)
    s_barrel = np.where(crosses_barrel, np.minimum(s_a, s_b), np.inf)

    # tracker endcap |z| = TRACKER_Z
    tan_lambda = cos_t / sin_t
    with np.errstate(divide="ignore", invalid="ignore"):
        s_endcap = (np.sign(tan_lambda) * TRACKER_Z - z0) / tan_lambda
    s_endcap = np.where((tan_lambda != 0.0) & np.isfinite(s_endcap) & (s_endcap > 0.0), s_endcap, np.inf)

    # a track that has not left the tracker within one transverse turn is a curler
    s_exit = np.minimum(s_barrel, s_endcap)
    s_exit = np.where(s_exit <= 2.0 * np.pi * radius, s_exit, np.inf)
    left_tracker = np.isfinite(s_exit)
    s_safe = np.where(left_tracker, s_exit, 0.0)

    alpha_exit = alpha0 - charge * s_safe / radius
    xe = cx + radius * np.cos(alpha_exit)
    ye = cy + radius * np.sin(alpha_exit)
    ze = z0 + s_safe * tan_lambda
    ux = sin_t * charge * np.sin(alpha_exit)
    uy = -sin_t * charge * np.cos(alpha_exit)
    uz = cos_t

    # straight line to the first ECAL surface
    qa = ux * ux + uy * uy
    qb = 2.0 * (xe * ux + ye * uy)
    qc = xe * xe + ye * ye - ECAL_BARREL_R**2
    disc = qb * qb - 4.0 * qa * qc
    with np.errstate(divide="ignore", invalid="ignore"):
        t_barrel = (-qb + np.sqrt(np.clip(disc, 0.0, None))) / (2.0 * qa)
    t_barrel = np.where((qa > 0.0) & (disc >= 0.0) & (t_barrel > 0.0), t_barrel, np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_endcap = (np.sign(uz) * ECAL_ENDCAP_Z - ze) / uz
    t_endcap = np.where((uz != 0.0) & np.isfinite(t_endcap) & (t_endcap > 0.0), t_endcap, np.inf)

    t_hit = np.minimum(t_barrel, t_endcap)
    reaches_ecal = left_tracker & np.isfinite(t_hit)
    t_safe = np.where(reaches_ecal, t_hit, 0.0)
    x = np.where(reaches_ecal, xe + t_safe * ux, 0.0)
    y = np.where(reaches_ecal, ye + t_safe * uy, 0.0)
    z = np.where(reaches_ecal, ze + t_safe * uz, 0.0)
    return x, y, z, reaches_ecal


def _impact_or_own(d0, z0, phi, theta, qop, own_eta, own_phi):
    """(eta, phi) of the ECAL impact point seen from the origin; own (eta, phi) where the helix does not reach it."""
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        x, y, z, reach = extrapolate_to_ecal(d0, z0, phi, theta, qop)
        r = np.hypot(x, y)
        eta = np.arcsinh(z / r)
    phi_i = np.arctan2(y, x)
    ok = reach & np.isfinite(eta) & np.isfinite(phi_i)
    return np.where(ok, eta, own_eta), np.where(ok, phi_i, own_phi)


def element_position(X):
    """(eta, phi) per element row, float64."""
    X = np.asarray(X)
    eta = X[:, 2].astype(np.float64)
    phi = np.arctan2(X[:, 3].astype(np.float64), X[:, 4].astype(np.float64))
    trk = X[:, 0] == 1
    if trk.any():
        # track columns 6-9 are d0, z0, theta, qop
        t = X[trk].astype(np.float64)
        eta[trk], phi[trk] = _impact_or_own(t[:, 6], t[:, 7], phi[trk], t[:, 8], t[:, 9], eta[trk], phi[trk])
    return eta, phi


def target_position(gen, gen_track, X, track_pos):
    """(eta, phi) per gen_target row. track_pos is element_position(X); gen_track indexes the track rows of X, -1 = none."""
    gen = np.asarray(gen)
    gen_track = np.asarray(gen_track).astype(np.int64)
    eta = gen[:, GEN_ETA].astype(np.float64)
    phi = np.arctan2(gen[:, GEN_SIN_PHI].astype(np.float64), gen[:, GEN_COS_PHI].astype(np.float64))
    charged = gen[:, GEN_CHARGE] != 0

    track_rows = np.flatnonzero(np.asarray(X)[:, 0] == 1)
    linked = charged & (gen_track >= 0)
    if np.any(gen_track >= len(track_rows)):
        raise ValueError(f"gen_target_track points past the {len(track_rows)} track rows")
    rows = track_rows[gen_track[linked]]
    eta[linked], phi[linked] = track_pos[0][rows], track_pos[1][rows]

    helix = charged & ~linked
    if helix.any():
        q = np.sign(gen[helix, GEN_CHARGE].astype(np.float64))
        pt = gen[helix, GEN_PT].astype(np.float64)
        theta = 2.0 * np.arctan(np.exp(-eta[helix]))
        with np.errstate(divide="ignore", over="ignore"):
            qop = q / (pt * np.cosh(eta[helix]))
        zeros = np.zeros(int(helix.sum()))
        eta[helix], phi[helix] = _impact_or_own(zeros, zeros, phi[helix], theta, qop, eta[helix], phi[helix])
    return eta, phi


def interior_id(eta, phi, eta_edges=ETA_EDGES, nphi=NPHI):
    """Crop whose interior holds each (eta, phi), in [0, n_crops)."""
    edges = np.asarray(eta_edges, dtype=np.float64)
    ie = np.searchsorted(edges[1:-1], np.asarray(eta, dtype=np.float64), side="right")
    ip = np.floor(np.mod(np.asarray(phi, dtype=np.float64) + math.pi, TWO_PI) / (TWO_PI / nphi)).astype(np.int64)
    return ie * nphi + np.clip(ip, 0, nphi - 1)


def in_crop(eta, phi, crop, guard, eta_edges=ETA_EDGES, nphi=NPHI):
    """Objects inside the crop's interior widened by guard."""
    ie, k = divmod(int(crop), nphi)
    lo, hi = eta_edges[ie], eta_edges[ie + 1]
    me = (eta >= lo - guard) & (eta < hi + guard)
    lo_phi = -math.pi + k * TWO_PI / nphi - guard
    return me & (np.mod(phi - lo_phi, TWO_PI) < TWO_PI / nphi + 2 * guard)


def choose_crops(seed, epoch, event_id, k=4, n=20):
    """k distinct crops of n for this (seed, epoch, event), independent of process, worker and visiting order."""
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(epoch), int(event_id)]))
    return sorted(int(c) for c in rng.choice(n, size=k, replace=False))


def positions(rec):
    """Element and target (eta, phi) of a record, computed once for all its crops."""
    epos = element_position(rec["X"])
    gen = rec.get("gen_target")
    if gen is None:
        return epos, None
    gen_track = rec.get("gen_target_track")
    if gen_track is None:
        gen_track = np.full(len(gen), -1, dtype=np.int64)
    return epos, target_position(gen, gen_track, rec["X"], epos)


def crop_record(rec, crop, guard, eta_edges=ETA_EDGES, nphi=NPHI, pos=None):
    """One crop of a record (None = the whole event), with elem_interior per kept element."""
    n_elem = len(rec["X"])
    if crop is None:
        out = dict(rec)
        out["elem_interior"] = np.ones(n_elem, dtype=bool)
        return out

    epos, tpos = positions(rec) if pos is None else pos
    interior = interior_id(*epos, eta_edges, nphi) == crop
    keep = interior | in_crop(*epos, crop, guard, eta_edges, nphi)
    out = dict(rec)
    for key in PER_ELEMENT_KEYS:
        if key in rec:
            out[key] = rec[key][keep]
    out["elem_interior"] = interior[keep]

    if tpos is not None:
        tkeep = interior_id(*tpos, eta_edges, nphi) == crop
        for key in PER_TARGET_KEYS:
            if key in rec:
                out[key] = rec[key][tkeep]
        if "gen_target_track" in rec:
            # remap to the kept track rows
            is_track = np.asarray(rec["X"])[:, 0] == 1
            new_index = np.full(int(is_track.sum()), -1, dtype=np.int64)
            kept_tracks = keep[is_track]
            new_index[kept_tracks] = np.arange(int(kept_tracks.sum()))
            old = np.asarray(rec["gen_target_track"]).astype(np.int64)[tkeep]
            out["gen_target_track"] = np.where(old >= 0, new_index[np.maximum(old, 0)], -1).astype(np.asarray(rec["gen_target_track"]).dtype)
    elif "ytarget_set" in rec:
        raise ValueError("cropping set targets needs gen_target positions")
    return out
