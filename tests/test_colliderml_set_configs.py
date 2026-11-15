from pathlib import Path

import pytest
import torch

from mlpf.conf import MLPFConfig
from mlpf.model.mlpf import MLPF
from mlpf.model.PFDataset import PFBatch, gen_target_to_set
from mlpf.model.training import model_step
from test_colliderml_crops import make_record

SPEC = str(Path(__file__).resolve().parents[1] / "particleflow_spec.yaml")
CONFIGS = ["pyg-colliderml-pu0-v20-set-hits", "pyg-colliderml-pu0-v20-set-hits-seedmax-dice", "pyg-colliderml-pu200-v22-set-hits-crops"]


@pytest.mark.parametrize("name", CONFIGS)
def test_set_configs_build_and_take_a_cpu_step(name):
    config = MLPFConfig.from_spec(SPEC, name, "colliderml")
    assert config.set_target_source == "gen_target" and config.input_dim == 18 and config.model.colliderml_input_features
    assert config.crop.enabled == name.endswith("crops") and config.colliderml_data.host_class_targets == name.endswith("crops")
    assert config.num_steps == 20000 and config.gradient_accumulation_steps == 4 and config.task_loss_weights.calibration_steps == 400
    assert config.optimizer.value == "adamw" and config.gradient_clip_norm == 1.0 and config.ntest == 0
    batch_size = config.train_dataset["colliderml"]["physical"].batch_size
    if name.endswith("crops"):
        assert batch_size == 2 and config.crop.guard == 0.2 and config.crop.crops_per_event == 4
    else:
        assert batch_size == 8
    decoder = config.model.set_decoder
    assert (decoder.calo_seed_local_max_radius is not None) == ("seedmax" in name)
    MLPF(config)

    # flash attention has no CPU backend: the smoke runs the same model on math attention
    d = config.model_dump()
    d["model"]["attention"] |= {"attention_type": "math", "use_jagged_attention": False}
    d["model"]["set_decoder"]["num_slots"] = 256
    torch.manual_seed(0)
    model = MLPF(MLPFConfig.model_validate(d))
    rec = make_record(seed=3, n_trk=20, n_cell=100, n_gen=30, eta_max=2.0)
    X = torch.zeros(1, len(rec["X"]), 18)
    X[0, :, :17] = torch.as_tensor(rec["X"])
    calo = X[0, :, 0] == 2
    X[0, calo, 10] = X[0, calo, 5]
    ytarget = torch.as_tensor(rec["ytarget"])[None]
    ytarget_set = torch.as_tensor(gen_target_to_set(rec["gen_target"]))[None]
    batch = PFBatch(X=X, ytarget=ytarget, ytarget_set=ytarget_set, source_id=torch.zeros(1, dtype=torch.long))
    total, losses, _, _, _, _ = model_step(batch, model, None, {k: 1.0 for k in ("pt", "eta", "sin_phi", "cos_phi", "energy")})
    assert torch.isfinite(total) and "Pileup" in losses and "HitGrouping" in losses
    total.backward()
