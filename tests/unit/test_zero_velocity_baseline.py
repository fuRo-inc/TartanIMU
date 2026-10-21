import csv
import json
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from tartan_imu.model.common.losses import single_head_velocity_loss
from tartan_imu.training.zero_velocity_baseline import metric_rows, summarize, save_metrics
from train import Trainer


@pytest.mark.parametrize("enabled,normalized", [(False, False), (True, False), (True, True)])
def test_metrics_and_batch_weighted_l1(enabled, normalized):
    cfg = dict(enabled=enabled, normalize_mean=normalized)
    target = torch.tensor([[0.1, 0.2, 0.3], [1., 2., 3.], [4., 5., 6.]])
    rows = torch.cat([metric_rows(torch.zeros_like(t), t, cfg) for t in target.split(2)])
    metrics = summarize(rows.sum(0), len(rows))
    assert metrics["model_mse"] == metrics["zero_mse"]
    assert metrics["model_rmse_3d"] == metrics["zero_rmse_3d"]
    assert metrics["model_velocity_l1_monitor"] == metrics["zero_velocity_l1"]
    assert metrics["skill_score_vs_zero"] == 0
    assert metrics["zero_mse"] == pytest.approx(target.square().mean().item())
    expected = sum(single_head_velocity_loss(torch.zeros_like(t), torch.zeros_like(t), t, cfg)["loss"].sum().item()
                   for t in target.split(2)) / target.numel()
    assert metrics["zero_velocity_l1"] == pytest.approx(expected)
    exact = summarize(metric_rows(target, target, cfg).sum(0), 3)
    assert exact["model_rmse_3d"] == 0
    assert exact["skill_score_vs_zero"] == 1


def test_zero_gt_json_csv_and_resume(tmp_path):
    rows = metric_rows(torch.ones(3, 3), torch.zeros(3, 3), {})
    metrics = summarize(rows.sum(0), 3)
    assert metrics["skill_score_vs_zero"] is None
    json.dumps(metrics, allow_nan=False)
    for epoch, split in [(1, "train"), (1, "validation"), (1, "train"), (2, "train")]:
        save_metrics(tmp_path, epoch, split, metrics)
    with (tmp_path / "zero_velocity_baseline_metrics.csv").open() as stream:
        saved = list(csv.DictReader(stream))
    assert len(saved) == 3
    assert saved[0]["skill_score_vs_zero"] == ""


class DogModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor([0.2, -0.1, 0.3]))

    def forward(self, feat, motion_type, **kwargs):
        return {"dog": feat + self.bias}


def make_trainer(tmp_path, enabled, weighted):
    cfg = {
        "data": {"velocity_target": "current", "use_local_coord": True, "imu_freq": 200},
        "model": {"pred_velocity": True},
        "model_param": {"output_dim": 3, "window_time": 0.5},
        "train": {
            "use_multi_gpu": False, "use_amp": False, "epochs": 1,
            "start_cov_epochs": 9999, "out_dir": str(tmp_path),
            "scheduler": {"factor": 0.1, "patience": 8},
            "active_heads": ["dog"], "predict_start": 0, "min_lr_stop": 0,
            "zero_velocity_baseline": {"enabled": enabled},
            "velocity_loss": {"speed_weighting": {"enabled": weighted, "normalize_mean": True}},
        },
    }
    model = DogModel()
    trainer = Trainer(SimpleNamespace(local_rank=0, log=False), cfg, model,
                      torch.optim.Adam(model.parameters(), lr=0.001))
    trainer.device = torch.device("cpu")
    trainer._params_printed = True
    trainer.save_checkpoint_strategy = lambda *args: None
    return trainer


@pytest.mark.parametrize("weighted", [False, True])
def test_real_train_validation_no_wandb_and_identical_updates(tmp_path, weighted):
    torch.manual_seed(15)
    features = torch.randn(5, 2, 3)
    targets = torch.randn(5, 2, 3)
    targets[:, 0] = 1000  # Earlier sequence target must not enter endpoint metrics.
    orientation = torch.randn(5, 4, 4)  # Must not rotate body velocity.
    loader = DataLoader(TensorDataset(features, targets, orientation, torch.full((5,), 2)), batch_size=3)
    disabled = make_trainer(tmp_path / "disabled", False, weighted)
    enabled = make_trainer(tmp_path / "enabled", True, weighted)
    disabled.train(loader, loader)
    enabled.train(loader, loader)
    assert torch.equal(disabled.model.bias, enabled.model.bias)
    left, right = disabled.optimizer.state_dict(), enabled.optimizer.state_dict()
    assert left["param_groups"] == right["param_groups"]
    for key in left["state"][0]:
        assert torch.equal(left["state"][0][key], right["state"][0][key])
    assert disabled.scheduler.state_dict() == enabled.scheduler.state_dict()
    assert disabled.best_val_loss == enabled.best_val_loss
    assert not (tmp_path / "disabled" / "zero_velocity_baseline_metrics.csv").exists()
    with (tmp_path / "enabled" / "zero_velocity_baseline_metrics.csv").open() as stream:
        saved = list(csv.DictReader(stream))
    assert {r["split"] for r in saved} == {"train", "validation"}
    expected_zero_mse = targets[:, -1].double().square().mean().item()
    for row in saved:
        assert int(row["sample_count"]) == 5
        assert float(row["zero_mse"]) == pytest.approx(expected_zero_mse)
    predictions = features[:, -1] + enabled.model.bias.detach()
    expected = summarize(metric_rows(predictions, targets[:, -1], {}).sum(0), 5)
    assert float(saved[1]["model_mse"]) == pytest.approx(expected["model_mse"])


def test_wandb_logs_all_metrics_and_undefined_skill(tmp_path, monkeypatch):
    trainer = make_trainer(tmp_path, True, False)
    trainer.log = True
    logged = {}
    monkeypatch.setattr("train.write_wandb", lambda name, value, epoch, rank: logged.update({name: value}))
    rows = metric_rows(torch.zeros(2, 3, requires_grad=True), torch.zeros(2, 3), {})
    assert not rows.requires_grad
    metrics = summarize(rows.sum(0), 2)
    trainer._report_zero_baseline(1, "train", metrics)
    assert len(logged) == 8
    assert logged["train/zero_velocity_baseline/skill_score_vs_zero"] is None
    json.dumps(logged, allow_nan=False)
