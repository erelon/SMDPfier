"""Default reward aggregators for SMDPfier."""

from __future__ import annotations

from typing import Callable


def sum_rewards(rewards: list[float]) -> float:
    """Sum all rewards in the list (default aggregator).

    Simple summation of all per-primitive rewards collected during
    option execution. This is the most common aggregation strategy.

    Args:
        rewards: List of per-primitive rewards from option execution

    Returns:
        Sum of all rewards

    Examples:
        >>> sum_rewards([1.0, -0.5, 2.0])
        2.5

        >>> sum_rewards([])  # Empty list
        0.0
    """
    return sum(rewards)


def mean_rewards(rewards: list[float]) -> float:
    """Compute mean of all rewards in the list.

    Averages the per-primitive rewards, which can help normalize
    rewards across options of different lengths.

    Args:
        rewards: List of per-primitive rewards from option execution

    Returns:
        Mean of all rewards, or 0.0 if list is empty

    Examples:
        >>> mean_rewards([1.0, -0.5, 2.0])
        0.8333333333333334

        >>> mean_rewards([])  # Empty list
        0.0
    """
    if not rewards:
        return 0.0
    return sum(rewards) / len(rewards)


def discounted_sum(gamma: float = 0.99) -> Callable[[list[float]], float]:
    """Create a reward aggregator that applies temporal discounting.

    Returns a function that computes the discounted sum of rewards,
    where earlier rewards are weighted more than later ones. This is
    NOT the same as SMDP discounting (which uses γ^{ticks}).

    Args:
        gamma: Discount factor (0 < gamma <= 1)

    Returns:
        Function that computes discounted sum of reward list

    Examples:
        >>> discount_fn = discounted_sum(gamma=0.9)
        >>> discount_fn([1.0, 1.0, 1.0])  # 1.0 + 0.9*1.0 + 0.81*1.0 = 2.71
        2.71

        >>> discount_fn([])  # Empty list
        0.0
    """
    if not (0 < gamma <= 1.0):
        raise ValueError(f"gamma must be in (0, 1], got {gamma}")

    def _discounted_sum(rewards: list[float]) -> float:
        """Apply temporal discounting to reward sequence."""
        if not rewards:
            return 0.0

        discounted_total = 0.0
        discount_factor = 1.0

        for reward in rewards:
            discounted_total += discount_factor * reward
            discount_factor *= gamma

        return discounted_total

    return _discounted_sum

