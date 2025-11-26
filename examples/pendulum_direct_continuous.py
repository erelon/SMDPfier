"""Pendulum example using direct interface with continuous actions."""

import argparse

import gymnasium as gym

from smdpfier import Option, SMDPfier
from smdpfier.defaults import discounted_sum


def create_pendulum_options() -> list[Option]:
    """Create a set of predefined continuous action options for Pendulum.

    Returns hand-crafted options representing different control strategies:
    - Oscillation patterns
    - Stabilization sequences
    - Exploration movements
    """
    options = [
        # Oscillation patterns
        Option(
            actions=[[1.0], [-1.0], [1.0], [-1.0]],
            name="oscillate-high",
            meta={"category": "oscillation", "amplitude": "high"},
        ),
        Option(
            actions=[[0.5], [-0.5], [0.5], [-0.5]],
            name="oscillate-medium",
            meta={"category": "oscillation", "amplitude": "medium"},
        ),
        Option(
            actions=[[0.2], [-0.2], [0.2]],
            name="oscillate-low",
            meta={"category": "oscillation", "amplitude": "low"},
        ),
        # Stabilization sequences
        Option(
            actions=[[0.8], [0.4], [0.1], [0.0]],
            name="stabilize-gradual",
            meta={"category": "stabilization", "strategy": "gradual"},
        ),
        Option(
            actions=[[0.0], [0.0], [0.0], [0.0], [0.0]],
            name="hold-steady",
            meta={"category": "stabilization", "strategy": "passive"},
        ),
        # Exploration movements
        Option(
            actions=[[-1.0], [1.0], [0.0]],
            name="explore-left-right",
            meta={"category": "exploration", "pattern": "lr"},
        ),
        Option(
            actions=[[0.7], [-0.3], [0.9], [-0.1]],
            name="explore-asymmetric",
            meta={"category": "exploration", "pattern": "asymmetric"},
        ),
    ]

    return options


def main(max_steps: int = 8) -> None:
    """Demonstrate SMDPfier with Pendulum using direct interface and continuous actions."""
    # Create base environment
    env = gym.make("Pendulum-v1")

    # Get predefined continuous options
    continuous_options = create_pendulum_options()

    # Create SMDPfier with direct interface (default reward aggregation = sum, but using discounted_sum here)
    smdp_env = SMDPfier(
        env,
        options_provider=continuous_options,  # Static continuous options
        reward_agg=discounted_sum(gamma=0.95),  # Discounted reward aggregation
        action_interface="direct",  # Pass Option objects directly
        availability_fn=None,  # No masking for continuous actions
        precheck=False,  # Skip precheck for continuous demo
        rng_seed=42,
    )

    print("=== Pendulum with SMDPfier (Direct Interface, Continuous Actions) ===")
    print(f"Original action space: {env.action_space}")
    print(f"SMDP action space: {smdp_env.action_space}")  # Should match original
    print(f"Number of predefined options: {len(continuous_options)}")

    # Show available options
    print("\nAvailable options:")
    for i, opt in enumerate(continuous_options):
        print(f"  {i}: {opt.name} (len={len(opt)}) - {opt.meta}")

    # Run episode
    obs, info = smdp_env.reset(seed=789)
    print(f"\nInitial observation: {obs}")

    total_reward = 0
    step_count = 0

    for episode_step in range(max_steps):  # Limit steps for demo
        # Choose option directly (in real RL, this would come from policy)
        option_to_execute = continuous_options[episode_step % len(continuous_options)]

        print(f"\nStep {episode_step + 1}: Executing '{option_to_execute.name}'")
        print(f"  Actions: {option_to_execute.actions}")

        # Execute option by passing Option object directly
        obs, reward, terminated, truncated, info = smdp_env.step(option_to_execute)

        # Extract SMDP info
        smdp_info = info["smdp"]
        print(f"  Executed {smdp_info['k_exec']}/{smdp_info['option']['len']} actions")
        print(f"  Per-step rewards: {[f'{r:.3f}' for r in smdp_info['rewards']]}")
        print(f"  Discounted reward: {reward:.3f}")
        print(f"  Duration: {smdp_info['duration']} ticks (= k_exec)")
        print(f"  Early termination: {smdp_info['terminated_early']}")
        print(f"  Final observation: [{obs[0]:.3f}, {obs[1]:.3f}, {obs[2]:.3f}]")

        total_reward += reward
        step_count += 1

        if terminated or truncated:
            print(f"\nEpisode ended after {step_count} SMDP steps")
            break

    print(f"\nTotal discounted reward: {total_reward:.3f}")

    env.close()
    print("Demo completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pendulum SMDPfier example with continuous actions")
    parser.add_argument("--max-steps", type=int, default=8,
                        help="Maximum number of SMDP steps to run")
    args = parser.parse_args()
    main(args.max_steps)


