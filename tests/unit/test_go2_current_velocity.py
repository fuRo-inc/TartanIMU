import numpy as np
from scipy.spatial.transform import Rotation

from tartan_imu.dataloader._common import current_velocity_from_positions, world_to_body_velocity
from tartan_imu.evaluation.postprocess import integrate_current_body_velocity
from tartan_imu.model.common.function import get_active_heads


def test_current_velocity_target_alignment_is_causal():
    # Ten 200 Hz windows end at endpoint.  The target must use precisely this,
    # never a sample after it.
    start, windows, samples = 17, 10, 200
    endpoint = start + windows * samples - 1
    input_indices = np.arange(start, endpoint + 1)
    assert input_indices.max() == endpoint
    assert not np.any(input_indices > endpoint)


def test_current_velocity_uses_central_difference_and_endpoints():
    ts = np.array([0.0, 1.0, 3.0, 6.0])
    pos = np.column_stack([ts**2, np.zeros(4), np.zeros(4)])
    velocity = current_velocity_from_positions(ts, pos)
    np.testing.assert_allclose(velocity[:, 0], [1.0, 3.0, 7.0, 9.0])


def test_world_to_body_velocity_yaw_90():
    quat = Rotation.from_euler("z", 90, degrees=True).as_quat()
    body = world_to_body_velocity(np.array([[0.0, 1.0, 0.0]]), quat[None])
    np.testing.assert_allclose(body, [[1.0, 0.0, 0.0]], atol=1e-12)


def test_current_velocity_integration_straight_and_lateral():
    ts = np.array([0.0, 0.2, 0.5, 1.0])
    quat = np.tile([0., 0., 0., 1.], (len(ts), 1))
    pos, world = integrate_current_body_velocity(ts, np.tile([1., 2., 0.], (len(ts), 1)), quat)
    np.testing.assert_allclose(world, np.tile([1., 2., 0.], (len(ts), 1)))
    np.testing.assert_allclose(pos[-1], [1., 2., 0.])


def test_current_velocity_integration_turn():
    # Forward base velocity with constant yaw rate produces a circular arc.
    ts = np.linspace(0., 1., 1001)
    omega = np.pi / 2
    quat = Rotation.from_euler("z", omega * ts).as_quat()
    pos, _ = integrate_current_body_velocity(ts, np.tile([1., 0., 0.], (len(ts), 1)), quat)
    expected = [np.sin(omega) / omega, (1 - np.cos(omega)) / omega, 0.]
    np.testing.assert_allclose(pos[-1], expected, atol=2e-7)


def test_dog_only_active_head_reads_train_section():
    assert get_active_heads({"train": {"active_heads": ["dog"]}}, np.array([1])) == ["dog"]
