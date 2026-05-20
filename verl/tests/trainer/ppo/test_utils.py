import pytest
from omegaconf import OmegaConf

from verl.trainer.ppo.utils import need_reference_policy


@pytest.mark.parametrize(
    "adv_estimator,eta,expected",
    [
        ("gae", 0.0, False),
        ("prm", 0.0, False),
        ("prm", 0.5, True),
    ],
)
def test_need_reference_policy_for_prm_eta(adv_estimator: str, eta: float, expected: bool):
    config = OmegaConf.create(
        {
            "algorithm": {"use_kl_in_reward": False, "adv_estimator": adv_estimator},
            "actor_rollout_ref": {"actor": {"use_kl_loss": False}},
            "prm": {"eta": eta},
        }
    )

    assert need_reference_policy(config) is expected
