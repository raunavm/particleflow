"""Scaled input features of ColliderML elements, for the input embedding only.

The ColliderML element columns (tracks: type, pt, eta, sin_phi, cos_phi, p, d0, z0, theta, qop, tanLambda, omega,
radiusOfInnermostHit, n_meas; calorimeter elements: type, et, eta, sin_phi, cos_phi, E, x, y, z, iTheta, E_ecal, E_hcal,
E_other, num_hits, sigma_x, sigma_y, sigma_z) mix millimeters and GeV. This module puts every column on a scale near 1
and adds Fourier features of eta, phi and the distance from the origin. It runs in fp32 under autocast. The raw
X_features still reach seeding, the heads and the element-type masks.

With use_layer, X column 17 holds the calorimeter layer of the element, scaled by the number of layers of the
calorimeter with more energy. With track_entry, the features end with the calorimeter entry direction of each track
(calo_entry.calo_entry_eta_phi) and the own direction of each calorimeter element.
"""

import math

import torch
from torch import nn

from mlpf.conf import Dataset
from mlpf.model.calo_entry import CALO_ENTRY_GEOMETRY_MM, calo_entry_eta_phi

HCAL_SATURATION_GEV = 0.6
ECAL_CLIP_GEV = 9.68
ECAL_LAYERS, HCAL_LAYERS = 48, 36


def _log(v, floor=1e-4):
    return torch.log(v.abs().clamp_min(floor)) / 3.0


class ColliderMLInputFeatures(nn.Module):
    NUM_BASE = 21
    NUM_BANDS = 8  # per coordinate: phi, eta, distance
    NUM_ENTRY = 3 + 2 * 2 * NUM_BANDS

    def __init__(self, use_layer=False, track_entry=False):
        super().__init__()
        self.use_layer = bool(use_layer)
        self.num_entry = self.NUM_ENTRY if track_entry else 0
        self.num_output_features = self.NUM_BASE + int(self.use_layer) + 2 * 3 * self.NUM_BANDS + self.num_entry
        bands = 2.0 ** torch.arange(1, self.NUM_BANDS + 1, dtype=torch.float32)
        self.register_buffer("phi_freq", bands, persistent=False)
        self.register_buffer("eta_freq", math.pi * bands / 2.5, persistent=False)
        self.register_buffer("radius_freq", math.pi * bands / 4.0, persistent=False)  # meters

    def forward(self, X_features):
        with torch.autocast(device_type=X_features.device.type, enabled=False):
            x = torch.nan_to_num(X_features.float(), nan=0.0, posinf=0.0, neginf=0.0)
            is_track, is_calo = x[..., 0] == 1, x[..., 0] == 2
            trk, cal = is_track.float(), is_calo.float()
            energy = x[..., 5].abs().clamp_min(1e-4)
            radius = torch.linalg.vector_norm(x[..., 6:9], dim=-1) / 1000.0 * cal
            phi = torch.atan2(x[..., 3], x[..., 4])
            # HCAL cell saturation and ECAL endcap clipping
            hcal_high = (x[..., 5] > HCAL_SATURATION_GEV) & (x[..., 11] > 0.5 * x[..., 5])
            ecal_clip = (x[..., 5] >= ECAL_CLIP_GEV) & (x[..., 10] > 0.5 * x[..., 5])

            def calo_or(c, t):
                return cal * c + trk * t

            columns = [
                trk,
                cal,
                _log(x[..., 1]),
                x[..., 2] / 3.0,
                x[..., 3],
                x[..., 4],
                _log(x[..., 5]),
                calo_or(x[..., 6] / 1000.0, torch.asinh(x[..., 6])),
                calo_or(x[..., 7] / 1000.0, x[..., 7] / 100.0),
                calo_or(x[..., 8] / 1000.0, x[..., 8] / math.pi),
                calo_or(x[..., 9] / math.pi, torch.asinh(x[..., 9])),
                calo_or(x[..., 10] / energy, x[..., 10] / 5.0),
                calo_or(x[..., 11] / energy, torch.asinh(x[..., 11] * 1000.0)),
                calo_or(x[..., 12] / energy, x[..., 12] / 100.0),
                calo_or(torch.log1p(x[..., 13].abs()) / 3.0, x[..., 13] / 20.0),
                cal * torch.log1p(x[..., 14].abs()),
                cal * torch.log1p(x[..., 15].abs()),
                cal * torch.log1p(x[..., 16].abs()),
                cal * hcal_high.float(),
                cal * ecal_clip.float(),
                radius,
            ]
            if self.use_layer:
                hcal = x[..., 11] > x[..., 10]
                columns.append(cal * torch.where(hcal, x[..., 17] / HCAL_LAYERS, x[..., 17] / ECAL_LAYERS))
            angles = torch.cat(
                [phi.unsqueeze(-1) * self.phi_freq, x[..., 2:3] * self.eta_freq, radius.unsqueeze(-1) * self.radius_freq],
                dim=-1,
            )
            out = torch.cat([torch.stack(columns, dim=-1), torch.sin(angles), torch.cos(angles)], dim=-1)
            if self.num_entry:
                # track columns 7, 10, 11 are z0, tanLambda, omega
                e_eta, e_phi = calo_entry_eta_phi(phi, x[..., 7], x[..., 10], x[..., 11], *CALO_ENTRY_GEOMETRY_MM[Dataset.COLLIDERML.value])
                e_eta, e_phi = torch.where(is_track, e_eta, x[..., 2]), torch.where(is_track, e_phi, phi)
                e_angles = torch.cat([e_phi.unsqueeze(-1) * self.phi_freq, e_eta.unsqueeze(-1) * self.eta_freq], dim=-1)
                entry = torch.stack([e_eta / 3.0, torch.sin(e_phi), torch.cos(e_phi)], dim=-1)
                out = torch.cat([out, entry, torch.sin(e_angles), torch.cos(e_angles)], dim=-1)
            valid = (is_track | is_calo).unsqueeze(-1)
            return torch.where(valid, out, torch.zeros_like(out))
