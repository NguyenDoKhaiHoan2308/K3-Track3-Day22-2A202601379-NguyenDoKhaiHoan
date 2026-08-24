from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _matching_finite_arrays(*values: FloatArray) -> tuple[FloatArray, ...]:
    arrays = tuple(np.asarray(value, dtype=np.float64) for value in values)
    if not arrays or arrays[0].size == 0:
        raise ValueError("loss inputs must not be empty")
    if any(array.shape != arrays[0].shape for array in arrays[1:]):
        raise ValueError("loss inputs must have matching shapes")
    if any(not np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("loss inputs must contain only finite values")
    return arrays


def dpo_loss(
    policy_chosen_logps: FloatArray,
    policy_rejected_logps: FloatArray,
    ref_chosen_logps: FloatArray,
    ref_rejected_logps: FloatArray,
    beta: float,
) -> float:
    """Compute batch DPO loss from sequence log probabilities.

    Compare the policy preference margin with the reference preference margin.
    """
    if not np.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be a positive finite number")
    policy_chosen, policy_rejected, ref_chosen, ref_rejected = _matching_finite_arrays(
        policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps
    )
    logits = beta * ((policy_chosen - policy_rejected) - (ref_chosen - ref_rejected))
    return float(np.mean(np.logaddexp(0.0, -logits)))


def orpo_loss(
    sft_nll: FloatArray,
    chosen_logps: FloatArray,
    rejected_logps: FloatArray,
    lambda_orpo: float,
) -> float:
    """Compute a simplified ORPO-style objective.

    Sequence log probabilities are converted to log-odds using a stable log(1-p).
    """
    if not np.isfinite(lambda_orpo) or lambda_orpo < 0:
        raise ValueError("lambda_orpo must be a non-negative finite number")
    nll, chosen, rejected = _matching_finite_arrays(sft_nll, chosen_logps, rejected_logps)
    if np.any(nll < 0):
        raise ValueError("sft_nll must be non-negative")
    if np.any(chosen > 0) or np.any(rejected > 0):
        raise ValueError("log probabilities must be less than or equal to zero")

    def log_one_minus_exp(log_probability: FloatArray) -> FloatArray:
        cutoff = -np.log(2.0)
        return np.where(
            log_probability < cutoff,
            np.log1p(-np.exp(log_probability)),
            np.log(-np.expm1(log_probability)),
        )

    with np.errstate(divide="ignore", invalid="ignore"):
        chosen_log_odds = chosen - log_one_minus_exp(chosen)
        rejected_log_odds = rejected - log_one_minus_exp(rejected)
        preference_loss = np.logaddexp(0.0, -(chosen_log_odds - rejected_log_odds))
    if np.any(np.isnan(preference_loss)):
        raise ValueError("chosen and rejected probabilities produce undefined odds")
    return float(np.mean(nll) + lambda_orpo * np.mean(preference_loss))
