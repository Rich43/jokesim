
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

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
- Personality-driven noob behavior
"""

NOOB_ARCHETYPES = [
    {"label": "Coder", "frag_bias": 0.7, "frag_variance": 0.2, "death_mult": 1.1},
    {"label": "Twitch Streamer", "frag_bias": 1.2, "frag_variance": 0.5, "death_mult": 1.5},
    {"label": "Tactician", "frag_bias": 0.5, "frag_variance": 0.1, "death_mult": 0.6},
    {"label": "Rager", "frag_bias": 1.0, "frag_variance": 0.6, "death_mult": 2.0},
    {"label": "Clueless Dad", "frag_bias": 0.3, "frag_variance": 0.05, "death_mult": 0.9},
    {"label": "Speedrunner", "frag_bias": 1.5, "frag_variance": 0.3, "death_mult": 1.3},
    {"label": "Strategist", "frag_bias": 0.6, "frag_variance": 0.1, "death_mult": 0.5},
    {"label": "AFK Guy", "frag_bias": 0.1, "frag_variance": 0.01, "death_mult": 0.1},
]

def simulate_frag_race(noob_count=146, match_duration=10, map_type="open", config=None):
    default_config = {
        'pro_base_fpm': 70,
        'noob_base_fpm': 0.3,
        'noob_fpm_std': 0.1,
        'pro_fpm_penalty': {'open': lambda n: 1 - min(0.4, n / 500), 'tight': lambda n: 1 - min(0.6, n / 300)},
        'noob_fpm_boost': {'open': lambda n: 1 + min(0.8, (n - 50) / 150), 'tight': lambda n: 1 + min(1.2, (n - 50) / 100)},
        'noob_collision_penalty': lambda n: max(0.5, 1 - (n - 100) / 400),
        'bfg_prob': 0.001,
        'bfg_min_frags': 3,
        'bfg_max_frags': 7,
        'frag_variance': 0.1,
        'respawn_delay': 0.025,
        'death_rate_pro': 0.2,
        'death_rate_noob': 0.8,
    }
    config = config or default_config

    if noob_count < 1 or match_duration <= 0:
        raise ValueError("noob_count and match_duration must be positive")
    if map_type not in config['pro_fpm_penalty']:
        raise ValueError("map_type must be 'open' or 'tight'")

    time_step = 1 / 60
    steps = int(match_duration / time_step)
    pro_frags = 0
    noob_frags = 0
    pro_active_time = 0
    noob_active_counts = [0] * noob_count
    timeline = []

    noob_personalities = [random.choice(NOOB_ARCHETYPES) for _ in range(noob_count)]
    noob_fpms = []
    noob_variances = []
    noob_death_rates = []

    for p in noob_personalities:
        fpm = max(0, np.random.normal(config['noob_base_fpm'] * p['frag_bias'], config['noob_fpm_std']))
        noob_fpms.append(fpm)
        noob_variances.append(p['frag_variance'])
        noob_death_rates.append(config['death_rate_noob'] * p['death_mult'])

    pro_respawn_timer = 0
    noob_respawn_timers = [0] * noob_count

    for i in range(steps):
        time_min = i * time_step
        pro_respawn_timer = max(0, pro_respawn_timer - time_step)
        for j in range(noob_count):
            noob_respawn_timers[j] = max(0, noob_respawn_timers[j] - time_step)

        active_noobs = sum(1 for t in noob_respawn_timers if t == 0)

        pro_fpm = config['pro_base_fpm'] * config['pro_fpm_penalty'][map_type](noob_count)
        pro_fpm *= random.gauss(1, config['frag_variance'])

        noob_fpm = sum(fpm for fpm, t in zip(noob_fpms, noob_respawn_timers) if t == 0)
        noob_fpm *= config['noob_fpm_boost'][map_type](noob_count) * config['noob_collision_penalty'](noob_count)

        variance_boost = sum(random.gauss(1, v) for v, t in zip(noob_variances, noob_respawn_timers) if t == 0) / (active_noobs or 1)
        noob_fpm *= variance_boost

        pro_step_frags = np.random.poisson(pro_fpm * time_step) if pro_respawn_timer == 0 else 0
        noob_step_frags = np.random.poisson(noob_fpm * time_step) if active_noobs > 0 else 0

        if random.random() < config['bfg_prob'] and active_noobs > 0:
            noob_step_frags += random.randint(config['bfg_min_frags'], config['bfg_max_frags'])

        pro_frags += pro_step_frags
        noob_frags += noob_step_frags

        pro_deaths = np.random.poisson(pro_step_frags * config['death_rate_pro']) if pro_respawn_timer == 0 else 0
        if pro_deaths > 0:
            pro_respawn_timer = config['respawn_delay']

        for j in range(noob_count):
            if noob_respawn_timers[j] == 0:
                noob_deaths = np.random.poisson(noob_fpms[j] * time_step * noob_death_rates[j])
                if noob_deaths > 0:
                    noob_respawn_timers[j] = config['respawn_delay']

        pro_active_time += time_step if pro_respawn_timer == 0 else 0
        for j in range(noob_count):
            noob_active_counts[j] += time_step if noob_respawn_timers[j] == 0 else 0

        timeline.append((time_min, pro_frags, noob_frags))

    return timeline, pro_frags, noob_frags, noob_personalities

def print_results(pro_frags, noob_frags, match_duration):
    winner = "PRO" if pro_frags > noob_frags else "BEGINNER TEAM"
    margin = abs(int(pro_frags - noob_frags))
    print(f"\n📊 Final Result ({match_duration}-minute match)")
    print(f"Pro Frags: {int(pro_frags)}")
    print(f"Beginner Team Frags: {int(noob_frags)}")
    print(f"🏆 Winner: {winner} (by {margin} frags)")

def plot_timeline(timeline, noob_count, map_type):
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
    noob_count = 140
    match_duration = 10
    map_type = "open"
    timeline, pro_frags, noob_frags, personalities = simulate_frag_race(noob_count, match_duration, map_type)
    print_results(pro_frags, noob_frags, match_duration)
    plot_timeline(timeline, noob_count, map_type)
    labels = [p['label'] for p in personalities]
    summary = Counter(labels)
    print("\n🧠 Noob Archetype Breakdown:")
    for label, count in summary.items():
        print(f"{label}: {count}")

if __name__ == "__main__":
    main()
