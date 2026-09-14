from types import SimpleNamespace

import pytest
import torch

from tartan_imu.config import configer


class _CaptureTrainer:
    def __init__(self, *args, **kwargs):
        self.__dict__.update(kwargs)


def _cfg():
    return {"data": {"velocity_target": "window_mean"}, "train": {
        "use_multi_gpu": False, "dog_head_init": "pretrained", "freeze_backbone": False,
        "optimizer": {"method": "Adam", "learning_rate": 0.01, "weight_decay": 0.0},
    }}


def _checkpoint(path, model, epoch=12):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    torch.save({"model_state_dict": model.state_dict(), "epoch": epoch,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": {"last_epoch": 7}, "scaler_state_dict": {"scale": 2.0},
                "trainer_state": {"best_train_loss": 1.0, "best_val_loss": 2.0}}, path)


def _build(monkeypatch, cfg, model, **paths):
    monkeypatch.setattr(configer.train, "Trainer", _CaptureTrainer)
    return configer.build_trainer(SimpleNamespace(log=False), cfg, model, **paths)


def test_pretrained_warm_start_resets_epoch(tmp_path, monkeypatch):
    source = torch.nn.Linear(2, 1)
    path = tmp_path / "source.pt"
    _checkpoint(path, source)
    trainer = _build(monkeypatch, _cfg(), torch.nn.Linear(2, 1), pretrained_path=str(path))
    assert trainer.start_epoch == 0
    assert trainer.resume_state == {}


def test_pretrained_warm_start_does_not_restore_optimizer(tmp_path, monkeypatch):
    source = torch.nn.Linear(2, 1)
    path = tmp_path / "source.pt"
    _checkpoint(path, source)
    trainer = _build(monkeypatch, _cfg(), torch.nn.Linear(2, 1), pretrained_path=str(path))
    assert trainer.optimizer.state == {}


def test_resume_restores_epoch(tmp_path, monkeypatch):
    source = torch.nn.Linear(2, 1)
    path = tmp_path / "resume.pt"
    _checkpoint(path, source)
    trainer = _build(monkeypatch, _cfg(), torch.nn.Linear(2, 1), resume_path=str(path))
    assert trainer.start_epoch == 12


def test_resume_restores_training_state(tmp_path, monkeypatch):
    source = torch.nn.Linear(2, 1)
    path = tmp_path / "resume.pt"
    _checkpoint(path, source)
    trainer = _build(monkeypatch, _cfg(), torch.nn.Linear(2, 1), resume_path=str(path))
    assert trainer.resume_state["best_train_loss"] == 1.0
    assert trainer.resume_state["scheduler_state_dict"]["last_epoch"] == 7
    assert trainer.optimizer.state


def test_checkpoint_and_resume_are_mutually_exclusive(monkeypatch):
    with pytest.raises(ValueError, match="cannot be used together"):
        _build(monkeypatch, _cfg(), torch.nn.Linear(2, 1), pretrained_path="a", resume_path="b")
