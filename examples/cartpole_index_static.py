"""CartPole example using index interface with static options."""

import argparse

import gymnasium as gym

from smdpfier import Option, SMDPfier
from smdpfier.defaults import ConstantOptionDuration, sum_rewards


def main(max_steps: int = 10) -> None:
    """Demonstrate SMDPfier with CartPole using static options and index interface."""
    # Create base environment
    env = gym.make("CartPole-v1")

    # Define static options (action sequences)
    static_options = [
        Option(actions=[0, 0, 1], name="left-left-right", meta={"category": "mixed"}),
        Option(actions=[1, 1, 0], name="right-right-left", meta={"category": "mixed"}),
        Option(actions=[0, 0, 0], name="left-triple", meta={"category": "directional"}),
        Option(
            actions=[1, 1, 1], name="right-triple", meta={"category": "directional"}
        ),
        Option(actions=[0, 1], name="left-right", meta={"category": "short"}),
        Option(actions=[1, 0], name="right-left", meta={"category": "short"}),
    ]

    # Create SMDPfier with static options
    smdp_env = SMDPfier(
        env,
        options_provider=static_options,  # Static sequence
        duration_fn=ConstantOptionDuration(10),  # 10 ticks per option
        reward_agg=sum_rewards,
        action_interface="index",  # Use discrete action indices
        max_options=len(static_options),  # All options fit
        precheck=False,  # No precheck for simplicity
        partial_duration_policy="proportional",
        rng_seed=42,
    )

    print("=== CartPole with SMDPfier (Index Interface, Static Options) ===")
    print(f"Original action space: {env.action_space}")
    print(f"SMDP action space: {smdp_env.action_space}")
    print(f"Available options: {len(static_options)}")

    # Run episode
    obs, info = smdp_env.reset(seed=123)
    print(f"\nInitial observation shape: {obs.shape}")

    total_reward = 0
    step_count = 0

    for episode_step in range(max_steps):  # Limit steps for demo
        # Choose option (in real RL, this would come from policy)
        available_mask = info.get("action_mask")
        if available_mask is not None:
            available_indices = [i for i, avail in enumerate(available_mask) if avail]
            option_idx = available_indices[episode_step % len(available_indices)]
        else:
            option_idx = episode_step % len(static_options)

        print(f"\nStep {episode_step + 1}: Choosing option {option_idx}")

        # Execute option
        obs, reward, terminated, truncated, info = smdp_env.step(option_idx)

        # Extract SMDP info
        smdp_info = info["smdp"]
        print(f"  Option: {smdp_info['option']['name']}")
        print(f"  Executed {smdp_info['k_exec']}/{smdp_info['option']['len']} actions")
        print(f"  Rewards: {smdp_info['rewards']}")
        print(f"  Aggregated reward: {reward}")
        print(
            f"  Duration: {smdp_info['duration_exec']}/{smdp_info['duration_planned']} ticks"
        )
        print(f"  Early termination: {smdp_info['terminated_early']}")

        total_reward += reward
        step_count += 1

        if terminated or truncated:
            print(f"\nEpisode ended after {step_count} SMDP steps")
            print(f"Total reward: {total_reward}")
            break

    env.close()
    print("\nDemo completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CartPole SMDPfier example")
    parser.add_argument("--max-steps", type=int, default=10,
                        help="Maximum number of SMDP steps to run")
    args = parser.parse_args()
    main(args.max_steps)
