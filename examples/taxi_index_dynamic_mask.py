"""Taxi example using index interface with dynamic options and action masking."""

import argparse

import gymnasium as gym

from smdpfier import Option, SMDPfier
from smdpfier.defaults import mean_rewards


def create_taxi_options(obs, info: dict) -> list[Option]:
    """Dynamic option provider for Taxi environment.

    Creates context-aware options based on current state:
    - Single primitive actions
    - Short 2-step sequences
    - Longer navigation sequences
    """
    options = []

    # Single primitive actions (always available)
    single_actions = [
        Option([0], "south", meta={"type": "primitive", "direction": "south"}),
        Option([1], "north", meta={"type": "primitive", "direction": "north"}),
        Option([2], "east", meta={"type": "primitive", "direction": "east"}),
        Option([3], "west", meta={"type": "primitive", "direction": "west"}),
        Option([4], "pickup", meta={"type": "primitive", "action": "pickup"}),
        Option([5], "dropoff", meta={"type": "primitive", "action": "dropoff"}),
    ]
    options.extend(single_actions)

    # Navigation sequences (2-step combinations)
    nav_sequences = [
        Option([0, 2], "south-east", meta={"type": "navigation", "pattern": "diagonal"}),
        Option([0, 3], "south-west", meta={"type": "navigation", "pattern": "diagonal"}),
        Option([1, 2], "north-east", meta={"type": "navigation", "pattern": "diagonal"}),
        Option([1, 3], "north-west", meta={"type": "navigation", "pattern": "diagonal"}),
        Option([2, 2], "east-east", meta={"type": "navigation", "pattern": "straight"}),
        Option([3, 3], "west-west", meta={"type": "navigation", "pattern": "straight"}),
    ]
    options.extend(nav_sequences)

    return options


def taxi_availability_function(obs) -> list[int]:
    """Determine which actions are available in Taxi environment.

    In Taxi, some actions may be invalid in certain states (e.g.,
    pickup/dropoff when not at passenger/destination location).
    """
    # Decode Taxi state from observation
    # Taxi state encoding: ((taxi_row * 5 + taxi_col) * 5 + passenger_location) * 4 + destination
    state = obs

    # Extract components
    state % 4
    state //= 4
    passenger_location = state % 5  # 0-3: specific locations, 4: in taxi
    state //= 5
    state % 5
    state //= 5

    # Movement actions (0-3) are typically always available
    # But we'll restrict some pickup/dropoff actions for masking demonstration
    available_actions = [0, 1, 2, 3]  # south, north, east, west always available

    # Add pickup if passenger is not in taxi and we're at passenger location
    if passenger_location != 4:  # passenger not in taxi
        # For simplicity, allow pickup about 70% of the time (demonstration purposes)
        # In practice, this would check if taxi is at passenger location
        if (state + obs) % 10 < 7:  # pseudo-random condition
            available_actions.append(4)

    # Add dropoff if passenger is in taxi
    if passenger_location == 4:  # passenger in taxi
        # Allow dropoff about 60% of the time (demonstration purposes)
        if (state + obs) % 10 < 6:
            available_actions.append(5)

    return available_actions


def main(max_steps: int = 15) -> None:
    """Demonstrate SMDPfier with Taxi using dynamic options, masking, and precheck."""
    # Create base environment
    env = gym.make("Taxi-v3")

    # Create SMDPfier with dynamic options (default reward aggregation = sum, but using mean here)
    smdp_env = SMDPfier(
        env,
        options_provider=create_taxi_options,  # Dynamic callable
        reward_agg=mean_rewards,  # Average rewards across actions
        action_interface="index",  # Use discrete action indices
        max_options=12,  # Limit number of dynamic options
        availability_fn=taxi_availability_function,  # Action masking
        precheck=True,  # Enable precheck validation
        rng_seed=42,
    )

    print("=== Taxi with SMDPfier (Index Interface, Dynamic Options, Masking) ===")
    print(f"Original action space: {env.action_space}")
    print(f"SMDP action space: {smdp_env.action_space}")
    print(f"Max options: {smdp_env.max_options}")
    print(f"Precheck enabled: {smdp_env._precheck}")

    # Run episode
    obs, info = smdp_env.reset(seed=456)
    print(f"\nInitial state: {obs}")

    total_reward = 0
    step_count = 0

    for episode_step in range(max_steps):  # Limit steps for demo
        # Get available options with masking
        smdp_info = info.get("smdp", {})
        available_mask = smdp_info.get("action_mask")
        num_dropped = smdp_info.get("num_dropped", 0)

        print(f"\nStep {episode_step + 1}:")
        print(f"  Available options mask: {available_mask}")
        print(f"  Options dropped due to overflow: {num_dropped}")

        # Choose available option
        if available_mask is not None:
            available_indices = [i for i, avail in enumerate(available_mask) if avail]
            if not available_indices:
                print("  No available options! Skipping step.")
                continue
            option_idx = available_indices[0]  # Choose first available
        else:
            option_idx = 0  # Fallback

        print(f"  Choosing option index: {option_idx}")

        try:
            # Execute option (may raise validation errors due to precheck)
            obs, reward, terminated, truncated, info = smdp_env.step(option_idx)

            # Extract SMDP info
            smdp_info = info["smdp"]
            print(f"  Executed option: {smdp_info['option']['name']}")
            print(f"  Steps: {smdp_info['k_exec']}/{smdp_info['option']['len']}")
            print(f"  Rewards: {smdp_info['rewards']}")
            print(f"  Mean reward: {reward:.3f}")
            print(f"  Duration: {smdp_info['duration']} ticks (= k_exec)")
            print(f"  Early termination: {smdp_info['terminated_early']}")

            total_reward += reward
            step_count += 1

        except Exception as e:
            print(f"  Option execution failed: {e}")
            continue

        if terminated or truncated:
            print(f"\nEpisode ended after {step_count} SMDP steps")
            print(f"Total reward: {total_reward}")
            break

    env.close()
    print("\nDemo completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Taxi SMDPfier example with masking")
    parser.add_argument("--max-steps", type=int, default=15,
                        help="Maximum number of SMDP steps to run")
    args = parser.parse_args()
    main(args.max_steps)
