import random
import numpy as np
import matplotlib.pyplot as plt

"""
Frag Race Simulator — Quake III Arena: Pro vs Pooled Noobs
-----------------------------------------------------------
Simulates a 10-minute deathmatch between one pro player and N beginners, with pooled beginner kills.
Compares total frags to find the tipping point where noobs match or beat the pro.
Visualizes frag accumulation over time.

Key Features:
- Discrete frags using Poisson distribution
- Configurable parameters via dictionary
- Input validation
- Modular design for simulation, results, and plotting
- Optional BFG spikes for noobs
"""

def simulate_frag_race(noob_count=146, match_duration=10, config=None):
    """Simulate a frag race between a pro and a team of noobs."""
    # Default configuration
    default_config = {
        'pro_base_fpm': 70,  # Pro frags per minute
        'noob_base_fpm': 0.3,  # Noob frags per minute (per player)
        'pro_fpm_penalty': lambda n: 1 - min(0.5, n / 400),  # Pro saturation penalty
        'noob_fpm_boost': lambda n: 1 + min(1.0, (n - 50) / 100),  # Noob spam boost
        'bfg_prob': 0.002,  # BFG spike probability per second
        'bfg_min_frags': 3,  # Min BFG frags
        'bfg_max_frags': 7,  # Max BFG frags
        'frag_variance': 0.1,  # Std dev for frag rate noise
    }
    config = config or default_config

    # Input validation
    if noob_count < 1:
        raise ValueError("noob_count must be positive")
    if match_duration <= 0:
        raise ValueError("match_duration must be positive")

    time_step = 1 / 60  # 1 second in minutes
    steps = int(match_duration / time_step)
    pro_frags = 0
    noob_frags = 0
    timeline = []

    for i in range(steps):
        time_min = i * time_step

        # Adjust frag rates with random noise
        pro_fpm = config['pro_base_fpm'] * config['pro_fpm_penalty'](noob_count)
        pro_fpm *= random.gauss(1, config['frag_variance'])
        noob_fpm = noob_count * config['noob_base_fpm'] * config['noob_fpm_boost'](noob_count)
        noob_fpm *= random.gauss(1, config['frag_variance'])

        # Discrete frags using Poisson distribution
        pro_step_frags = np.random.poisson(pro_fpm * time_step)
        noob_step_frags = np.random.poisson(noob_fpm * time_step)

        # BFG spike
        if random.random() < config['bfg_prob']:
            noob_step_frags += random.randint(config['bfg_min_frags'], config['bfg_max_frags'])

        pro_frags += pro_step_frags
        noob_frags += noob_step_frags
        timeline.append((time_min, pro_frags, noob_frags))

    return timeline, pro_frags, noob_frags

def print_results(pro_frags, noob_frags, match_duration):
    """Print simulation results."""
    winner = "PRO" if pro_frags > noob_frags else "BEGINNER TEAM"
    margin = abs(int(pro_frags - noob_frags))
    print(f"\n📊 Final Result ({match_duration}-minute match)")
    print(f"Pro Frags: {int(pro_frags)}")
    print(f"Beginner Team Frags: {int(noob_frags)}")
    print(f"🏆 Winner: {winner} (by {margin} frags)")

def plot_timeline(timeline, noob_count):
    """Plot frag accumulation over time."""
    times = [t[0] for t in timeline]
    pro_line = [t[1] for t in timeline]
    noob_line = [t[2] for t in timeline]

    plt.figure(figsize=(12, 6))
    plt.plot(times, pro_line, label="Pro", linewidth=2, color='#1f77b4')
    plt.plot(times, noob_line, label="Beginner Team", linewidth=2, color='#ff7f0e')
    plt.title(f"Frag Race Simulation — {noob_count} Noobs vs 1 Pro")
    plt.xlabel("Time (minutes)")
    plt.ylabel("Frags")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    noob_count = 105
    match_duration = 10
    timeline, pro_frags, noob_frags = simulate_frag_race(noob_count, match_duration)
    print_results(pro_frags, noob_frags, match_duration)
    plot_timeline(timeline, noob_count)

if __name__ == "__main__":
    main()
