import random
import matplotlib.pyplot as plt

"""
Frag Race Simulator — Quake III Arena: Pro vs Pooled Noobs
-----------------------------------------------------------

Project Summary:
This simulation models a 10-minute deathmatch between one professional Quake III player 
and a group of N beginner players, where all beginner kills are pooled into a single score. 
The purpose is to identify the "tipping point" — the number of noobs required to match or 
beat the pro purely in kill count, ignoring deaths.

Design Goals:
- Compare total frags (kills) between the pro and beginner team.
- Model frag rate using configurable parameters and probabilistic BFG frag bursts.
- Visualize the match as a time-series graph showing frag accumulation per side.
- Tune balance variables like frag/min rates, spawn saturation penalties, and spam overlap effects.
- Support high-resolution simulation (1-second ticks) for smoother analysis.

Key Features:
- Adjustable beginner count (default: 146)
- 1-second simulation granularity (600 ticks over 10 minutes)
- Configurable frag rates:
    - Pro: high accuracy, suffers under saturation
    - Noobs: low individual skill, but benefit from mass spam and random BFG usage
- Optional BFG frag spikes for beginner team (random event)
- Outputs winner, final scores, frag margin
- Graphs frag accumulation over time using matplotlib

Usage:
- Run as a standalone script to simulate a single match.
- Modify `noob_count`, `pro_base_fpm`, `noob_base_fpm`, etc. for balance testing.
- Optional: integrate into a loop to test across a range of noob counts to identify equilibrium.

Author: ChatGPT for R
Date: 2025
License: MIT

"""


def simulate_frag_race(noob_count=146, match_duration=10):
    time_step = 1 / 60  # 1 second (since time is in minutes)
    steps = int(match_duration / time_step)

    pro_base_fpm = 70  # base frags per minute for the pro
    noob_base_fpm = 0.3  # average frags per noob per minute

    pro_fpm_penalty = lambda n: 1 - min(0.5, n / 400)  # map saturation penalty
    noob_fpm_boost = lambda n: 1 + min(1.0, (n - 50) / 100)  # spam overlap benefit

    pro_frags = 0
    noob_frags = 0
    timeline = []

    for i in range(steps):
        time_min = i * time_step

        # Adjust frag rates
        pro_fpm = pro_base_fpm * pro_fpm_penalty(noob_count)
        noob_fpm = noob_count * noob_base_fpm * noob_fpm_boost(noob_count)

        # Frags per this time step
        pro_step_frags = pro_fpm * time_step
        noob_step_frags = noob_fpm * time_step

        # Occasional BFG bonus spike for noobs
        if random.random() < 0.01:  # 1% chance per second
            noob_step_frags += random.randint(3, 7)

        # Accumulate
        pro_frags += pro_step_frags
        noob_frags += noob_step_frags

        timeline.append((time_min, pro_frags, noob_frags))

    # Final Result
    winner = "PRO" if pro_frags > noob_frags else "BEGINNER TEAM"
    margin = abs(int(pro_frags - noob_frags))

    print(f"\n📊 Final Result (10-minute match)")
    print(f"Pro Frags: {int(pro_frags)}")
    print(f"Beginner Team Frags: {int(noob_frags)}")
    print(f"🏆 Winner: {winner} (by {margin} frags)")

    # Plotting
    times = [t[0] for t in timeline]
    pro_line = [t[1] for t in timeline]
    noob_line = [t[2] for t in timeline]

    plt.figure(figsize=(12, 6))
    plt.plot(times, pro_line, label="Pro", linewidth=2)
    plt.plot(times, noob_line, label="Beginner Team", linewidth=2)
    plt.title(f"Frag Race Simulation — {noob_count} Noobs vs 1 Pro (1s tick)")
    plt.xlabel("Time (minutes)")
    plt.ylabel("Frags")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 🟢 RUN THE SIMULATOR
simulate_frag_race(noob_count=104)
