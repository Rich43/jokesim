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

def simulate_frag_race(noob_count=146, match_duration=10, map_type="open", config=None):
    """Simulate a Quake III deathmatch with deaths and map effects."""
    # Default configuration
    default_config = {
        'pro_base_fpm': 70,
        'noob_base_fpm': 0.3,
        'noob_fpm_std': 0.1,  # Std dev for noob skill variation
        'pro_fpm_penalty': {'open': lambda n: 1 - min(0.4, n / 500), 'tight': lambda n: 1 - min(0.6, n / 300)},
        'noob_fpm_boost': {'open': lambda n: 1 + min(0.8, (n - 50) / 150), 'tight': lambda n: 1 + min(1.2, (n - 50) / 100)},
        'noob_collision_penalty': lambda n: max(0.5, 1 - (n - 100) / 400),  # Penalty for high noob counts
        'bfg_prob': 0.001,  # BFG spawn every ~1000 seconds (rare)
        'bfg_min_frags': 3,
        'bfg_max_frags': 7,
        'frag_variance': 0.1,
        'respawn_delay': 0.025,  # 1.5 seconds in minutes
        'death_rate_pro': 0.2,  # Pro dies 20% as often as they frag
        'death_rate_noob': 0.8,  # Noobs die 80% as often as they frag
    }
    config = config or default_config

    # Input validation
    if noob_count < 1 or match_duration <= 0:
        raise ValueError("noob_count and match_duration must be positive")
    if map_type not in config['pro_fpm_penalty']:
        raise ValueError("map_type must be 'open' or 'tight'")

    time_step = 1 / 60  # 1 second in minutes
    steps = int(match_duration / time_step)
    pro_frags = 0
    noob_frags = 0
    pro_active_time = 0  # Time pro is alive
    noob_active_counts = [0] * noob_count  # Time each noob is alive
    timeline = []

    # Initialize noob skill variation
    noob_fpms = np.random.normal(config['noob_base_fpm'], config['noob_fpm_std'], noob_count)
    noob_fpms = np.clip(noob_fpms, 0, None)  # No negative frag rates

    pro_respawn_timer = 0
    noob_respawn_timers = [0] * noob_count

    for i in range(steps):
        time_min = i * time_step

        # Update respawn timers
        pro_respawn_timer = max(0, pro_respawn_timer - time_step)
        for j in range(noob_count):
            noob_respawn_timers[j] = max(0, noob_respawn_timers[j] - time_step)

        # Count active noobs
        active_noobs = sum(1 for t in noob_respawn_timers if t == 0)

        # Adjust frag rates
        pro_fpm = config['pro_base_fpm'] * config['pro_fpm_penalty'][map_type](noob_count)
        pro_fpm *= random.gauss(1, config['frag_variance'])
        noob_fpm = sum(fpm for fpm, t in zip(noob_fpms, noob_respawn_timers) if t == 0)
        noob_fpm *= config['noob_fpm_boost'][map_type](noob_count) * config['noob_collision_penalty'](noob_count)
        noob_fpm *= random.gauss(1, config['frag_variance'])

        # Calculate frags and deaths
        pro_step_frags = np.random.poisson(pro_fpm * time_step) if pro_respawn_timer == 0 else 0
        noob_step_frags = np.random.poisson(noob_fpm * time_step) if active_noobs > 0 else 0

        # BFG spike
        if random.random() < config['bfg_prob'] and active_noobs > 0:
            noob_step_frags += random.randint(config['bfg_min_frags'], config['bfg_max_frags'])

        # Update frags
        pro_frags += pro_step_frags
        noob_frags += noob_step_frags

        # Simulate deaths
        pro_deaths = np.random.poisson(pro_step_frags * config['death_rate_pro']) if pro_respawn_timer == 0 else 0
        if pro_deaths > 0:
            pro_respawn_timer = config['respawn_delay']
        for j in range(noob_count):
            if noob_respawn_timers[j] == 0:
                noob_deaths = np.random.poisson(noob_fpms[j] * time_step * config['death_rate_noob'])
                if noob_deaths > 0:
                    noob_respawn_timers[j] = config['respawn_delay']

        # Track active time
        pro_active_time += time_step if pro_respawn_timer == 0 else 0
        for j in range(noob_count):
            noob_active_counts[j] += time_step if noob_respawn_timers[j] == 0 else 0

        timeline.append((time_min, pro_frags, noob_frags))

    # Adjust frags based on active time (optional for further realism)
    return timeline, pro_frags, noob_frags

def print_results(pro_frags, noob_frags, match_duration):
    """Print simulation results."""
    winner = "PRO" if pro_frags > noob_frags else "BEGINNER TEAM"
    margin = abs(int(pro_frags - noob_frags))
    print(f"\n📊 Final Result ({match_duration}-minute match)")
    print(f"Pro Frags: {int(pro_frags)}")
    print(f"Beginner Team Frags: {int(noob_frags)}")
    print(f"🏆 Winner: {winner} (by {margin} frags)")

def plot_timeline(timeline, noob_count, map_type):
    """Plot frag accumulation over time."""
    times = [t[0] for t in timeline]
    pro_line = [t[1] for t in timeline]
    noob_line = [t[2] for t in timeline]

    plt.figure(figsize=(12, 6))
    plt.plot(times, pro_line, label="Pro", linewidth=2, color='#1f77b4')
    plt.plot(times, noob_line, label="Beginner Team", linewidth=2, color='#ff7f0e')
    plt.title(f"Frag Race — {noob_count} Noobs vs 1 Pro ({map_type} map)")
    plt.xlabel("Time (minutes)")
    plt.ylabel("Frags")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    noob_count = 110
    match_duration = 10
    map_type = "open"  # Try "open" or "tight"
    timeline, pro_frags, noob_frags = simulate_frag_race(noob_count, match_duration, map_type)
    print_results(pro_frags, noob_frags, match_duration)
    plot_timeline(timeline, noob_count, map_type)

if __name__ == "__main__":
    main()
