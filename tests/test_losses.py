import numpy as np
import pytest

from preference_lab.losses import dpo_loss, orpo_loss


def test_dpo_loss() -> None:
    loss = dpo_loss(
        np.array([-0.5]), np.array([-1.5]), np.array([-0.6]), np.array([-1.0]), beta=0.1
    )
    assert loss == pytest.approx(np.logaddexp(0.0, -0.06))


def test_dpo_loss_is_stable_for_extreme_margin() -> None:
    loss = dpo_loss(
        np.array([-1.0]), np.array([-100_000.0]), np.array([-1.0]), np.array([-2.0]), beta=1.0
    )
    assert np.isfinite(loss)
    assert loss == pytest.approx(0.0)


def test_dpo_loss_is_finite_with_equal_reference_logps() -> None:
    loss = dpo_loss(
        np.array([-0.5]),
        np.array([-1.5]),
        np.array([-1.0]),
        np.array([-1.0]),
        beta=0.1,
    )
    assert loss == pytest.approx(np.logaddexp(0.0, -0.1))
    assert np.isfinite(loss)


def test_orpo_loss() -> None:
    loss = orpo_loss(np.array([1.0]), np.array([-0.5]), np.array([-1.5]), lambda_orpo=0.1)
    assert loss > 1.0
    assert np.isfinite(loss)
