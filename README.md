# SC2-LSTM-Bot

A reinforcement learning bot for StarCraft II using LSTM networks with temporal attention mechanisms. Built with PyTorch and PySC2.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![PySC2](https://img.shields.io/badge/PySC2-4.0+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## Features

- **LSTM with Temporal Attention**: 4-layer LSTM (1024 hidden units) with 8-head attention mechanism for long-term strategic memory
- **Tiered Curriculum Learning**: 3-tier reward system (Economy → Army → Combat) with fast-track graduation for high performers
- **Map-Agnostic Design**: Coordinate transformation system ensures consistent behavior regardless of spawn position
- **Distributed Training**: 8 parallel workers with centralized GPU server for efficient training
- **Complete Terran Tech Tree**: 58 discrete actions covering buildings, units, upgrades, and 6 strategic combat macros
- **PPO Training**: Proximal Policy Optimization with entropy bonus for stable learning
- **Comprehensive Logging**: Real-time metrics tracking with visualization dashboard

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      Training Architecture                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐                  │
│   │ Worker 0 │    │ Worker 1 │    │ Worker N │                  │
│   │  SC2 Env │    │  SC2 Env │    │  SC2 Env │                  │
│   └────┬─────┘    └────┬─────┘    └────┬─────┘                  │
│        │               │               │                         │
│        └───────────────┼───────────────┘                         │
│                        │                                         │
│                        ▼                                         │
│              ┌─────────────────┐                                 │
│              │   GPU Server    │                                 │
│              │  ┌───────────┐  │                                 │
│              │  │   LSTM    │  │                                 │
│              │  │ + Attn    │  │                                 │
│              │  └───────────┘  │                                 │
│              │  ┌───────────┐  │                                 │
│              │  │    PPO    │  │                                 │
│              │  │  Trainer  │  │                                 │
│              │  └───────────┘  │                                 │
│              └─────────────────┘                                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Requirements

### Hardware
- **GPU**: NVIDIA RTX 3090 (24GB VRAM) or equivalent
- **CPU**: AMD Ryzen 5900X or equivalent (12+ cores recommended for parallel workers)
- **RAM**: 64GB DDR4 recommended
- **Storage**: SSD with 50GB+ free space

### Software
- Python 3.8+
- StarCraft II (retail or free version)
- CUDA 11.0+

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/sc2-lstm-bot.git
cd sc2-lstm-bot
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Install StarCraft II**
   - Download from [Blizzard](https://starcraft2.com/)
   - Install maps: Download [ladder maps](https://github.com/Blizzard/s2client-proto#map-packs) and extract to `StarCraftII/Maps/`

5. **Set environment variable** (Linux/Mac)
```bash
export SC2PATH="/path/to/StarCraftII"
```

## Quick Start

### Training

```bash
# Start distributed training with 8 workers
python parallel_main_memory.py
```

Training will:
- Launch 8 parallel SC2 instances
- Train for 1,250 episodes per worker (10,000 total)
- Save checkpoints every 100 updates
- Generate metrics logs for visualization

### Visualization

```bash
# Generate training dashboard
python visualize_training.py
```

This creates `training_dashboard.png` with:
- Win rate over time
- Game length distribution
- Unit/building composition analysis
- Building placement heatmaps
- Action distribution

## Project Structure

```
sc2-lstm-bot/
├── parallel_main_memory.py    # Main entry point for distributed training
├── gpu_server_memory.py       # Centralized GPU server with PPO training
├── worker_memory.py           # Game worker process
├── worker_rewards.py          # Tiered reward system with curriculum learning
├── lstm_memory.py             # LSTM network with temporal attention
├── game_perception.py         # State encoding (78 features)
├── coordinate_transform.py    # Map-agnostic coordinate system
├── base_building_system.py    # Expansion tracking and gating
├── terran_actions.py          # Action registry and combat macros
├── terran_buildings.py        # Building construction functions
├── terran_units.py            # Unit training and abilities
├── terran_legal_actions_logic.py  # Action preconditions
├── terran_helper.py           # Utility functions
├── metrics_logger.py          # Training metrics collection
├── visualize_training.py      # Dashboard generation
└── simple64.png               # Map visualization asset
```

## Configuration

### Training Parameters

Edit `parallel_main_memory.py`:

```python
NUM_WORKERS = 8              # Parallel SC2 instances
EPISODES_PER_WORKER = 1250   # Episodes per worker
```

Edit `gpu_server_memory.py`:

```python
hidden_size = 1024           # LSTM hidden units
num_lstm_layers = 3          # LSTM depth
num_attention_heads = 8      # Attention heads
lr = 3e-4                    # Learning rate
sequence_length = 128        # Training sequence length
batch_size = 64              # Training batch size
ppo_clip = 0.2               # PPO clipping ratio
entropy_coef = 0.01          # Entropy bonus
```

Edit `worker_memory.py`:

```python
max_steps = 10_000           # Max steps per episode
epsilon_start = 0.3          # Initial exploration rate
epsilon_end = 0.05           # Final exploration rate
epsilon_decay = 0.995        # Epsilon decay per episode
```

### State Space

The bot perceives 78 features organized into:

| Category | Features | Description |
|----------|----------|-------------|
| Game Time | 3 | Minutes, phase, game loop |
| Resources | 4 | Minerals, gas, supply used/cap |
| Supply State | 2 | Available, full flags |
| Spatial | 8 | Base, army, enemy positions |
| Economy | 5 | CC count, SCVs, refineries |
| Buildings | 18 | All Terran structure counts |
| Army | 18 | All Terran unit counts |
| Army State | 4 | Supply, health ratio, combat |
| Enemy Info | 10 | Race, units, buildings |
| Upgrades | 6 | Research status flags |

### Action Space

58 discrete actions:

| Category | Actions |
|----------|---------|
| Basic | do_nothing |
| Workers | send_idle_workers_to_mine/gas, train_worker |
| Buildings | supply_depot, barracks, factory, starport, command_center, etc. |
| Addons | techlab/reactor for barracks, factory, starport |
| Units | marine, marauder, hellion, siege_tank, medivac, etc. |
| Abilities | call_down_mule, scanner_sweep |
| Research | barracks/factory/starport/engineering_bay upgrades |
| Combat | attack_aggressor, attack_siege, attack_support, attack_harass, retreat_to_base, final_push |

## Combat Macros

The bot uses 6 strategic combat macros instead of low-level attack commands:

| Macro | Description | Units Used |
|-------|-------------|------------|
| `attack_aggressor` | Frontline assault | Marines, Marauders, Hellbats, Thors, BCs |
| `attack_siege` | Area control positioning | Siege Tanks, Liberators, Widow Mines |
| `attack_support` | Support positioning | Medivacs, Ravens |
| `attack_harass` | Fast harassment | Reapers, Hellions, Banshees |
| `retreat_to_base` | Retreat to nearest CC | All army |
| `final_push` | All-in attack | All army |

## Coordinate System

The coordinate transformation ensures map-agnostic behavior:

```
Top-Left Spawn          Bottom-Right Spawn
┌───────────────┐       ┌───────────────┐
│ CC            │       │            CC │
│   ↘           │  ───► │           ↙   │
│      Enemy    │       │    Enemy      │
└───────────────┘       └───────────────┘
        │                       │
        └───────────┬───────────┘
                    ▼
          Normalized [0,1] Space
          ┌───────────────┐
          │ CC (0,0)      │
          │   ↘           │
          │      Enemy    │
          └───────────────┘
```

## Base Building System

Expansion is gated by base completion requirements:

| Tier | Buildings | Required |
|------|-----------|----------|
| Anchor | Command Center/Orbital/Planetary | 1 |
| Production | Barracks, Factory, Starport | 2 |
| Accessory | Supply Depot, Engineering Bay, etc. | 3 |

A base must have 1/2/3 of these tiers before expanding to the next location.

## Interruption Handling

Training is designed to survive interruptions:

- **Ctrl+C**: Graceful shutdown, saves model
- **Crashes**: Workers can be restarted, GPU server preserves state
- **Checkpoints**: Saved every 100 updates

## Troubleshooting

### Common Issues

**SC2 won't launch**
```bash
# Check SC2PATH
echo $SC2PATH
# Should point to StarCraftII directory
```

**Out of VRAM**
```python
# Reduce in gpu_server_memory.py:
hidden_size = 512  # Down from 1024
batch_size = 32    # Down from 64
```

**Workers crashing**
```bash
# Check crash logs
cat crash_log_worker_*.txt
```

**Slow training**
```python
# Reduce workers in parallel_main_memory.py:
NUM_WORKERS = 4  # Down from 8
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [DeepMind PySC2](https://github.com/deepmind/pysc2) - StarCraft II Learning Environment
- [PyTorch](https://pytorch.org/) - Deep Learning Framework
- [Blizzard Entertainment](https://www.blizzard.com/) - StarCraft II

## Citation

If you use this project in your research, please cite:

```bibtex
@misc{sc2lstmbot2025,
  author = {Anthony Galindo},
  title = {SC2-LSTM-Bot: Reinforcement Learning for StarCraft II},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/aerynaaronson/starcraft-rl-agent}
}
```