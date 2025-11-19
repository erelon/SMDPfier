#!/usr/bin/env python
"""Quick verification script for rate-based default reward."""

import gymnasium as gym
from smdpfier import SMDPfier, Option
from smdpfier.defaults import ConstantActionDuration, sum_rewards


def test_rate_default():
    """Verify that default reward is rate-based."""
    print("=" * 60)
    print("Testing Rate-Based Default Reward")
    print("=" * 60)

    # Create two environments with same option but different durations
    env1 = gym.make("CartPole-v1")
    env2 = gym.make("CartPole-v1")

    option = [Option([0, 1], "test-option")]

    # Fast: 1 tick per action = 2 total
    smdp_fast = SMDPfier(
        env1,
        options_provider=option,
        duration_fn=ConstantActionDuration(1),
    )

    # Slow: 5 ticks per action = 10 total
    smdp_slow = SMDPfier(
        env2,
        options_provider=option,
        duration_fn=ConstantActionDuration(5),
    )

    # Execute with same seed
    obs1, info1 = smdp_fast.reset(seed=42)
    obs2, info2 = smdp_slow.reset(seed=42)

    obs1, reward_fast, term1, trunc1, info1 = smdp_fast.step(0)
    obs2, reward_slow, term2, trunc2, info2 = smdp_slow.step(0)

    # Get info
    duration_fast = info1["smdp"]["duration_exec"]
    duration_slow = info2["smdp"]["duration_exec"]
    rewards_fast = info1["smdp"]["rewards"]
    rewards_slow = info2["smdp"]["rewards"]

    print(f"\nFast Option:")
    print(f"  Duration: {duration_fast} ticks")
    print(f"  Primitive rewards: {rewards_fast}")
    print(f"  Sum: {sum(rewards_fast)}")
    print(f"  Macro reward (rate): {reward_fast}")
    print(f"  Expected: {sum(rewards_fast) / duration_fast}")

    print(f"\nSlow Option:")
    print(f"  Duration: {duration_slow} ticks")
    print(f"  Primitive rewards: {rewards_slow}")
    print(f"  Sum: {sum(rewards_slow)}")
    print(f"  Macro reward (rate): {reward_slow}")
    print(f"  Expected: {sum(rewards_slow) / duration_slow}")

    # Verify rate formula
    assert abs(reward_fast - sum(rewards_fast) / duration_fast) < 1e-6
    assert abs(reward_slow - sum(rewards_slow) / duration_slow) < 1e-6

    # Verify faster option gets higher reward (assuming positive rewards)
    if sum(rewards_fast) > 0:
        assert reward_fast > reward_slow, "Faster option should yield higher reward!"
        print(f"\n✅ SUCCESS: Fast option has higher reward ({reward_fast:.4f} > {reward_slow:.4f})")

    env1.close()
    env2.close()


def test_legacy_sum():
    """Verify that sum_rewards still works for backward compatibility."""
    print("\n" + "=" * 60)
    print("Testing Legacy sum_rewards Compatibility")
    print("=" * 60)

    env = gym.make("CartPole-v1")
    option = [Option([0, 1], "test-option")]

    smdp_sum = SMDPfier(
        env,
        options_provider=option,
        duration_fn=ConstantActionDuration(10),
        reward_agg=sum_rewards,  # Explicit legacy
    )

    obs, info = smdp_sum.reset(seed=42)
    obs, reward, term, trunc, info = smdp_sum.step(0)

    rewards = info["smdp"]["rewards"]
    duration = info["smdp"]["duration_exec"]

    print(f"\nWith sum_rewards (legacy):")
    print(f"  Duration: {duration} ticks")
    print(f"  Primitive rewards: {rewards}")
    print(f"  Macro reward: {reward}")
    print(f"  Expected (sum): {sum(rewards)}")

    # Should equal sum, not rate
    assert abs(reward - sum(rewards)) < 1e-6
    print(f"\n✅ SUCCESS: Legacy sum_rewards works ({reward} == {sum(rewards)})")

    env.close()


if __name__ == "__main__":
    test_rate_default()
    test_legacy_sum()
    print("\n" + "=" * 60)
    print("All verification tests passed! ✅")
    print("=" * 60)

