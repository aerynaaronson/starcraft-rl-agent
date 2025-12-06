# StarCraft II Reinforcement Learning Agent

**LSTM-based RL agent that learns macro strategy from pure win/loss signals — zero hardcoding, emergent behavior.**

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PySC2](https://img.shields.io/badge/PySC2-latest-green.svg)](https://github.com/deepmind/pysc2)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This project explores **pure reinforcement learning** in StarCraft II without relying on hardcoded strategies, build orders, or scripted behaviors. The agent learns Terran macro strategy from scratch using only win/loss outcomes (with time penalties to encourage faster wins).

Starting with random actions, the agent discovers:
- Economic expansion patterns
- Tech tree progression
- Building placement strategies  
- Aggressive timing attacks
- Redundant base construction for resilience

**Current Performance:** 34%+ winrate against Very Easy AI after 150 episodes, trending upward.

---

## Key Features

### 🧠 LSTM-Based Policy Network
- Maintains memory across game timesteps (unlike feedforward networks)
- Learns temporal dependencies: "I built a barracks 2 minutes ago, now I should make marines"
- 128 hidden units with proper forget/input/output gates
- Learns coordinate mapping for building placement per action type

### 🎯 Dynamic Legal Action System (**Key Innovation**)
The biggest challenge in SC2 RL is the massive action space. This system solves it elegantly:

- **Function availability checker** gates actions based on:
  - Current resources (minerals, vespene)
  - Tech requirements (e.g., Factory requires Barracks)
  - Unit existence (can't train marines without a Barracks)
  - Supply availability
  
- Network only sees **valid actions at each timestep**
- Prevents wasted exploration on impossible actions
- Action space grows naturally as tech tree is unlocked

Example: Early game the network can only choose from `[train_scv, build_supply_depot, build_barracks]`. After building a Factory, `[train_hellion, train_siege_tank, build_armory]` become available.

### 🎮 Game State Perception Module
Extracts 35-dimensional state vector including:
- Economy: minerals, vespene, supply used/cap
- Units: SCVs, marines, marauders, factory units, starport units
- Buildings: command centers, production facilities, tech structures, add-ons
- Tech status: techlabs, reactors, armories, fusion cores

### 📊 Comprehensive Metrics & Visualization
- **Real-time logging:** Every action, building placement, economy snapshot
- **Matplotlib dashboards:** 
  - Rolling winrate curves
  - Win vs loss comparisons (unit counts, building counts, game length)
  - Building placement heatmaps
  - Economy progression over time
  - Action distribution analysis
- Saves to `.jsonl` format for easy post-analysis

### 🚀 Emergent Behaviors Observed
After 150 episodes, the agent has learned:
- **Radical expansion:** Consistently builds 2-3+ command centers
- **Relentless aggression:** Attacks early and often (time penalty effect)
- **Backup bases:** Accidentally achieves redundancy by over-expanding
- **Tech progression:** Reliably reaches Factory + TechLab
- **Fast wins:** Winning games average 1400-2000 steps vs 3000-4000 for losses

---

## Architecture Details

### Network Design
```
State (35 dims) → LSTM (128 hidden) → Output Layer (35 actions)
                                    ↓
                              Coordinate Weights (per action, 2D)
```

**Why LSTM?** StarCraft requires temporal reasoning:
- "I'm supply blocked" → "I built a depot 30 seconds ago" → "It should finish soon"
- "I have 400 minerals" → "I've been saving" → "Time to expand"

### Training Details
- **Reward:** +100 for win, -100 for loss, with time penalty (encourages faster wins)
- **Epsilon-greedy exploration:** Starts at 1.0, decays to 0.05 over 1000 episodes
- **Learning rate:** 0.01
- **No replay buffer:** Updates happen end-of-episode based on outcome
- **Model saves:** Every 50 episodes

### Action Execution
One hardcoded macro for testing: `attack_move` selects all military units and attacks nearest enemy. This proves the concept works before adding per-unit micro. All other actions (build, train, expand) are pure NN decisions.

---

## Project Structure

```
starcraft-rl-agent/
│
├── game_perception.py          # Extracts game state into feature vector
├── terran_functions.py         # Wraps PySC2 actions into callable functions
├── terran_legal_actions_logic.py  # Determines which actions are currently valid
├── lstm_action_network.py      # LSTM policy network implementation
├── metrics_logger.py           # Logs all game data for analysis
├── analyzer.py                 # Generates visualization dashboards
├── main.py                     # Training loop
│
├── proof_of_concept/
│   └── hardcode_terran_bot.py  # Early DQN prototype (80% hardcoded, had memory leak)
│
├── models/
│   ├── terran_model.pkl        # Saved every 50 episodes
│   └── terran_model_final.pkl  # Final trained model
│
└── results/
    ├── training_metrics.jsonl  # Per-episode data (actions, placements, outcomes)
    └── training_dashboard.png  # Generated visualization
```

---

## Installation

### 1. Install StarCraft II
Download the free version:
```bash
# Follow official PySC2 instructions:
# https://github.com/deepmind/pysc2#quick-start-guide
```

### 2. Install Python Dependencies
```bash
pip install pysc2 numpy matplotlib
```

No PyTorch, TensorFlow, or GPU required. Pure NumPy implementation.

---

## Usage

### Train the Agent
```bash
python main.py
```

Training parameters (edit in `main.py`):
- `episodes = 1000` - Total training episodes
- `epsilon_decay = 0.995` - Exploration decay rate
- `learning_rate = 0.01` - Network learning rate
- `Difficulty.very_easy` - Opponent difficulty

### Generate Analysis Dashboard
After training completes:
```bash
python analyzer.py
```

Creates `training_dashboard.png` with:
- Winrate progression
- Strategy comparisons
- Building heatmaps
- Economy curves

### Watch the Agent Play
Set `visualize=True` in `main.py` to watch games in real-time (slower).

---

## Experimental Design

Currently running comparative study:

| Map Size | Difficulty | Episodes | Status |
|----------|-----------|----------|---------|
| Simple64 | Very Easy | 1000 | In Progress (Ep 156) |
| Simple64 | Medium | 1000 | Planned |
| Large Map | Very Easy | 1000 | Planned |
| Large Map | Hard | 1000 | Planned |

**Research Questions:**
1. Does map complexity bottleneck learning independent of opponent skill?
2. Do different environments produce different emergent strategies?
3. How does curriculum learning (difficulty scaling) affect RL performance?

---

## Results (Episode 156 / 1000)

**Winrate:** 34% (up from ~10% at episode 50)  
**Fastest Win:** 1374 steps (~61 seconds game time)  
**Average Win:** ~1900 steps  
**Average Loss:** ~3400 steps  

**Observed Strategies:**
- Consistent 2-3 base expansion
- Factory + TechLab progression
- Aggressive marine/hellion/siege tank compositions
- Rarely reaches Starport tech (wins before needing it)

---

## Why This Approach?

**Most SC2 RL projects rely on:**
- Pretrained models (AlphaStar)
- Hardcoded build orders
- Scripted micro
- Imitation learning from replays
- Heavily shaped reward functions

**This project uses:**
- ✅ Pure win/loss signal (+ time penalty)
- ✅ Zero hardcoding (except one attack macro for testing)
- ✅ No replay data
- ✅ Simple LSTM architecture
- ✅ Runs on CPU

**Goal:** Prove that a simple system can discover complex strategy through pure trial and error.

---

## Future Work

- [ ] Complete 8000-episode comparative study
- [ ] Add per-unit micro control (remove attack-all macro)
- [ ] Test cross-map transfer learning
- [ ] Implement self-play training
- [ ] Scale to larger maps and harder difficulties
- [ ] Add other races (Zerg, Protoss)

---

## Evolution from Prototype

Early prototype (`proof_of_concept/hardcode_terran_bot.py`):
- PyTorch DQN with CNN
- 85% GPU utilization, 100% CPU
- Heavily scripted action queues
- Memory leak crashed at episode 80
- Reward shaping for every action

Current version:
- Pure NumPy LSTM
- ~25% CPU, no GPU needed
- Function availability system (no scripts)
- Stable training for 150+ episodes
- Pure win/loss learning

**Key insight:** Simpler is better. The complex version was overengineered.

---

## Contributing

This is a research/educational project. Feel free to:
- Open issues for bugs or questions
- Submit PRs for improvements
- Fork for your own experiments

---

## License

MIT License - feel free to use for research, education, or commercial projects.

---

## Acknowledgments

- DeepMind for PySC2
- Blizzard for StarCraft II API
- The RL research community for inspiration

---

## Contact

**Anthony Galindo**  
📧 anthonywgalindo@gmail.com  
💼 [LinkedIn](https://linkedin.com/in/awgalindo)  
🐙 [GitHub](https://github.com/aerynaaronson)

If you use this project in your research, please cite or link back to this repository.