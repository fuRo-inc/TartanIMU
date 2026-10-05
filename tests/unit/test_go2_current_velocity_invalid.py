import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tartan_imu.dataloader.dataset_AirLab import AirLabNPZSequence, BasicSequenceData


def _write_npz(path, jump=False, velocity_body=None, velocity_world=None, quat=None):
    n, hz = 150, 10.0
    ts = np.arange(n) / hz
    pos = np.column_stack((ts, np.zeros(n), np.zeros(n)))
    if jump:
        pos[60, 0] += 100.0
    data = {
        "retargetted_ts": ts,
        "retargetted_imu": np.column_stack((np.tile([0., 0., 9.81], (n, 1)), np.zeros((n, 3)))),
        "retargetted_pos": pos,
        "retargetted_quat": np.tile([0., 0., 0., 1.], (n, 1)) if quat is None else quat,
    }
    if velocity_body is not None:
        data["retargetted_velocity_body"] = velocity_body
    if velocity_world is not None:
        data["root_linear_velocity"] = velocity_world
    np.savez(path, **data)


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


def test_direct_body_velocity_is_authoritative_and_not_re_rotated(tmp_path):
    path = tmp_path / "dog_direct_body.npz"
    n = 150
    direct_body = np.tile([1.0, 0.0, 0.0], (n, 1))
    # This makes position differentiation disagree radically with the direct GT.
    quat = np.tile(Rotation.from_euler("z", 90, degrees=True).as_quat(), (n, 1))
    _write_npz(
        path,
        jump=True,
        velocity_body=direct_body,
        velocity_world=np.tile([0.0, 3.0, 0.0], (n, 1)),
        quat=quat,
    )

    seq = AirLabNPZSequence(str(path), 10, 10, verbose=False, use_local_coord=True,
                             velocity_target="current", current_velocity_max_speed=10)

    assert seq.valid
    assert seq.current_velocity_source == "direct_body"
    # It is neither derived from the position jump nor rotated by the yaw.
    np.testing.assert_allclose(seq.targets, direct_body)
    assert seq.current_velocity_valid.all()


def test_direct_body_nonfinite_endpoint_is_excluded_without_fallback(tmp_path):
    path = tmp_path / "dog_direct_body_nan.npz"
    direct_body = np.tile([12.0, 0.0, 0.0], (150, 1))
    direct_body[59] = np.nan
    _write_npz(path, velocity_body=direct_body)

    seq = AirLabNPZSequence(str(path), 10, 10, verbose=False, use_local_coord=True,
                             velocity_target="current", current_velocity_max_speed=10)
    assert seq.current_velocity_source == "direct_body"
    assert not seq.current_velocity_valid[59]
    # Direct velocities above the legacy fallback threshold remain valid.
    assert seq.current_velocity_valid[58]

    data = BasicSequenceData(_cfg(), [str(path)], mode="train")
    endpoints = [j + data.seq_len * data.window_size - 1 for _, j, _ in data.index_map[0]]
    assert 59 not in endpoints


def test_direct_body_velocity_shape_mismatch_raises(tmp_path):
    path = tmp_path / "dog_direct_body_bad_shape.npz"
    _write_npz(path, velocity_body=np.zeros((149, 3)))

    with pytest.raises(ValueError, match="retargetted_velocity_body must have shape"):
        AirLabNPZSequence(str(path), 10, 10, verbose=False, use_local_coord=True,
                           velocity_target="current")


def test_legacy_direct_world_velocity_is_rotated_to_body_frame(tmp_path):
    path = tmp_path / "dog_direct_world.npz"
    n = 150
    world = np.tile([0.0, 1.0, 0.0], (n, 1))
    quat = np.tile(Rotation.from_euler("z", 90, degrees=True).as_quat(), (n, 1))
    _write_npz(path, velocity_world=world, quat=quat)

    seq = AirLabNPZSequence(str(path), 10, 10, verbose=False, use_local_coord=True,
                             velocity_target="current", current_velocity_max_speed=10)
    assert seq.current_velocity_source == "direct_world"
    np.testing.assert_allclose(seq.targets, np.tile([1.0, 0.0, 0.0], (n, 1)), atol=1e-12)
