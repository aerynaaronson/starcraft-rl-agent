# StarCraft II Reinforcement Learning Agent

**LSTM-based RL agent that learns macro strategy from pure win/loss signals — zero hardcoding, emergent behavior.**

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PySC2](https://img.shields.io/badge/PySC2-latest-green.svg)](https://github.com/deepmind/pysc2)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This project explores **pure reinforcement learning** in StarCraft II without relying on hardcoded strategies, build orders, or scripted behaviors. The agent learns Terran macro strategy from scratch using only win/loss outcomes (pure +100/-100 reward, no time penalties or reward shaping).

Starting with random actions, the agent discovers:
- Economic expansion patterns
- Tech tree progression (Marines → Factory → Starport → Battlecruisers)
- Building placement strategies  
- Multiple winning strategies (fast timing attacks AND late-game compositions)
- Redundant base construction for economic resilience

**Current Performance:** 40%+ winrate against Very Easy AI after 252 episodes, still climbing.

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
- Units: SCVs, marines, marauders, factory units, starport units, battlecruisers
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

**After 252 episodes, the agent has discovered TWO distinct strategies:**

#### Strategy 1: Fast Timing Attack (Primary, ~90% of wins)
- **Radical expansion:** Consistently builds 2-3+ command centers
- **Relentless aggression:** Attacks early and maintains pressure
- **Backup bases:** Accidentally achieves redundancy by over-expanding
- **Tech progression:** Reliably reaches Factory + TechLab
- **Composition:** Marine/Hellion/Siege Tank
- **Game length:** 1200-2200 steps
- **Fastest win recorded:** 1193 steps (Episode 166)

#### Strategy 2: Late Game Air Superiority (Emerging, ~10% of wins)
- **Full tech tree:** Factory → Starport → Fusion Core
- **Advanced units:** Thors AND Battlecruisers
- **Extended economy:** Multi-base operation
- **Game length:** 3000-3500 steps
- **First observed:** Episode 250

**Key Finding:** Agent discovers strategic diversity despite no explicit reward for tech advancement. Both strategies achieve +100 reward, proving the system can learn multiple paths to victory.

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
- **Reward:** Pure +100 for win, -100 for loss (NO time penalty, NO reward shaping)
- **Epsilon-greedy exploration:** Starts at 1.0, decays to 0.05 over 1000 episodes
- **Learning rate:** 0.01
- **No replay buffer:** Updates happen end-of-episode based on outcome
- **Model saves:** Every 50 episodes
- **Current epsilon (ep 252):** 0.29 (~29% exploration)

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
├── EXPERIMENT_LOG.md           # Comprehensive findings and observations
│
├── proof_of_concept/
│   └── hardcode_terran_bot.py  # Early DQN prototype (abandoned)
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
pip install -r requirements.txt
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

Currently running comparative study across map sizes and difficulty levels:

| Map Size | Difficulty | Episodes | Status |
|----------|-----------|----------|---------|
| Simple64 | Very Easy | 1000 | **In Progress (252/1000)** |
| Simple64 | Easy | 1000 | Planned |
| Simple64 | Medium | 1000 | Planned |
| Simple64 | Hard | 1000 | Planned |
| Large Map | Very Easy | 1000 | Planned |
| Large Map | Easy | 1000 | Planned |
| Large Map | Medium | 1000 | Planned |
| Large Map | Hard | 1000 | Planned |

**Total:** 8,000 episodes across 8 experimental conditions

**Research Questions:**
1. Does map complexity bottleneck learning independent of opponent skill?
2. Do different environments produce different emergent strategies?
3. How does curriculum learning (difficulty scaling) affect RL performance?
4. Can strategic knowledge transfer across map sizes?
5. What causes sample efficiency bias toward fast strategies?

---

## Results (Episode 252 / 1000)

### Performance Metrics
- **Winrate:** 40%+ (up from ~10% at episode 50)  
- **Fastest Win:** 1193 steps (Episode 166)  
- **Typical Fast Win:** 1400-2000 steps  
- **Typical Slow Win:** 3000-3500 steps
- **Average Loss:** ~3400 steps  

### Strategic Evolution
- **Episodes 1-50:** Random exploration, learning not to die (~10-15% winrate)
- **Episodes 50-100:** Initial pattern recognition (~20-25% winrate)
- **Episodes 100-200:** Strategy emergence - marine/factory timing attacks (~30-40% winrate)
- **Episodes 200-252:** Strategic diversity - battlecruiser composition discovered (~40%+ winrate)

### Observed Unit Compositions
**Winning Games:**
- Fast wins: Marines, Hellions, Siege Tanks (Factory tech)
- Slow wins: Marines, Thors, Battlecruisers (Full tech tree)

**Tech Tree Progression:**
- Barracks: 100% of games
- Factory + TechLab: ~85% of games
- Starport: ~40% of games
- Starport + TechLab + Fusion Core: ~10% of games

---

## Why This Approach?

**Most SC2 RL projects rely on:**
- Pretrained models (AlphaStar)
- Hardcoded build orders
- Scripted micro
- Imitation learning from replays
- Heavily shaped reward functions

**This project uses:**
- ✅ Pure win/loss signal (no time penalty, no reward shaping)
- ✅ Zero hardcoding (except one attack macro for testing)
- ✅ No replay data
- ✅ Simple LSTM architecture
- ✅ Runs on CPU (~25% utilization)

**Goal:** Prove that a simple system can discover complex strategy through pure trial and error.

---

## Key Findings

### 1. Sample Efficiency Bias
Fast strategies (1500 steps) get more training samples per hour than slow strategies (3500 steps). This creates natural bias toward timing attacks even with neutral rewards.

**10 fast wins = 10 gradient updates in 4 hours**  
**5 slow wins = 5 gradient updates in 4 hours**

Despite this, the agent still discovers battlecruiser compositions, proving LSTM can explore full strategy space.

### 2. Strategic Diversity Without Reward Shaping
Episode 250 proves complex, multi-stage strategies (Battlecruisers require 5+ tech buildings) emerge naturally from pure win/loss learning.

### 3. Function Availability Enables Efficient Exploration
Dynamic action filtering prevents wasting compute on invalid actions. Early game: 3-5 legal actions. Late game: 25-35 legal actions.

### 4. Architectural Simplification
Pure NumPy LSTM (25% CPU) outperforms PyTorch DQN+CNN (85% GPU, 100% CPU) through better problem abstraction. Simpler is better.

### 5. Accidental Emergent Behaviors
"More command centers = good economy" accidentally creates backup bases for resilience. Not programmed, discovered through trial and error.

---

## Future Work

- [ ] Complete Simple64 Very Easy (1000 episodes) - **In Progress**
- [ ] Scale difficulty (Easy → Medium → Hard)
- [ ] Test large map performance
- [ ] Add per-unit micro control (remove attack-all macro)
- [ ] Fix coordinate mapping bug (bottom 20% of map)
- [ ] Test cross-map transfer learning
- [ ] Implement self-play training
- [ ] Add other races (Zerg, Protoss)

---

## Evolution from Prototype

### Early Prototype (Abandoned)
- PyTorch DQN with CNN architecture
- 85% GPU utilization, 100% CPU utilization
- Heavily scripted action queues
- Reward shaping for every action
- Memory leak crashed at episode 80
- Complex but underperformed

### Current Version
- Pure NumPy LSTM implementation
- ~25% CPU, ~10% GPU (minimal)
- Zero scripted behaviors (except one attack macro)
- Pure win/loss reward
- Stable training 252+ episodes
- Simple but effective

**Key Insight:** Clean abstractions (function availability system) > complex architectures.

---

## Documentation

See `EXPERIMENT_LOG.md` for comprehensive findings, observations, and technical details.

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

---

## Citation

```bibtex
@misc{galindo2024sc2rl,
  author = {Galindo, Anthony},
  title = {StarCraft II Reinforcement Learning Agent: Emergent Strategy from Pure Win/Loss Learning},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/aerynaaronson/starcraft-rl-agent}
}
```