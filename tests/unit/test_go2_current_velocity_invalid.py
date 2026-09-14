import numpy as np

from tartan_imu.dataloader.dataset_AirLab import AirLabNPZSequence, BasicSequenceData


def _write_npz(path, jump=False):
    n, hz = 150, 10.0
    ts = np.arange(n) / hz
    pos = np.column_stack((ts, np.zeros(n), np.zeros(n)))
    if jump:
        pos[60, 0] += 100.0
    np.savez(
        path,
        retargetted_ts=ts,
        retargetted_imu=np.column_stack((np.tile([0., 0., 9.81], (n, 1)), np.zeros((n, 3)))),
        retargetted_pos=pos,
        retargetted_quat=np.tile([0., 0., 0., 1.], (n, 1)),
    )


def _cfg():
    return {
        "model_param": {"window_time": 1.0, "past_time": 0, "future_time": 0},
        "data": {"imu_freq": 10.0, "sample_freq": 10.0, "use_local_coord": True,
                 "velocity_target": "current", "current_velocity_max_speed": 10.0},
        "train": {"seq_len": 2, "add_noise": False},
    }


def test_current_velocity_normal_fallback_is_kept(tmp_path):
    path = tmp_path / "dog_normal.npz"
    _write_npz(path)
    seq = AirLabNPZSequence(str(path), 10, 10, verbose=False, use_local_coord=True,
                             velocity_target="current", current_velocity_max_speed=10)
    assert seq.valid
    assert seq.current_velocity_source == "position_difference"
    assert seq.current_velocity_valid.all()
    np.testing.assert_allclose(seq.targets[:, 0], 1.0)


def test_position_jump_endpoint_is_excluded_not_clipped(tmp_path):
    path = tmp_path / "dog_jump.npz"
    _write_npz(path, jump=True)
    seq = AirLabNPZSequence(str(path), 10, 10, verbose=False, use_local_coord=True,
                             velocity_target="current", current_velocity_max_speed=10)
    # The raw fallback label preserves the anomaly; it is not turned into a
    # plausible-looking 10 m/s teacher target.
    assert np.linalg.norm(seq.targets[59]) > 10
    assert not seq.current_velocity_valid[59]

    data = BasicSequenceData(_cfg(), [str(path)], mode="train")
    endpoints = [j + data.seq_len * data.window_size - 1 for _, j, _ in data.index_map[0]]
    assert 59 not in endpoints
    assert all(np.linalg.norm(data.targets[0][endpoint]) <= 10 for endpoint in endpoints)
