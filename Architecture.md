# Architecture Documentation

This document provides detailed technical documentation of the SC2-LSTM-Bot architecture.

## Table of Contents

1. [Neural Network Architecture](#neural-network-architecture)
2. [State Encoding](#state-encoding)
3. [Action System](#action-system)
4. [Coordinate Transformation](#coordinate-transformation)
5. [Distributed Training](#distributed-training)
6. [Reward System](#reward-system)
7. [Base Building System](#base-building-system)

---

## Neural Network Architecture

### LSTMActionNetwork

The core neural network is defined in `lstm_memory.py`:

```
Input (78 features)
       │
       ▼
┌─────────────────┐
│  Input Project  │  Linear(78 → 1024)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Positional    │  Sinusoidal encoding (max 15000 steps)
│    Encoding     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   4-Layer LSTM  │  hidden_size=1024, bidirectional=False
│                 │  dropout=0.1
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Layer Norm 1   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Temporal      │  8 attention heads
│   Attention     │  Attends over episode memory buffer
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Layer Norm 2   │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌───────┐ ┌───────┐
│Action │ │Coord  │
│Head   │ │Head   │
│(58)   │ │(2)    │
└───────┘ └───────┘
```

### Key Components

#### Positional Encoding
```python
pe[:, 0::2] = sin(position / 10000^(2i/d_model))
pe[:, 1::2] = cos(position / 10000^(2i/d_model))
```
Enables the network to understand temporal position within an episode.

#### Temporal Attention
Multi-head attention mechanism that allows the current state to attend to all previous states in the episode:

```python
Q = W_q @ current_state      # Query from current state
K = W_k @ episode_memory     # Keys from all past states
V = W_v @ episode_memory     # Values from all past states
attention = softmax(QK^T / √d_k) @ V
```

#### Episode Memory Buffer
Rolling buffer storing up to 10,000 LSTM outputs for attention:
- Clears at episode start
- Accumulates during play
- Enables long-term strategic reasoning

### Model Statistics
- **Parameters**: ~50M
- **VRAM Usage**: ~4-6GB during training
- **Inference Time**: ~5ms per step

---

## State Encoding

The state encoder in `game_perception.py` produces a 78-dimensional feature vector:

### Feature Groups

#### Game Time (3 features)
| Index | Feature | Normalization |
|-------|---------|---------------|
| 0 | Game minutes | ÷30 |
| 1 | Game phase | 0-1 (early/mid/late) |
| 2 | Game loop | ÷20000 |

#### Resources (4 features)
| Index | Feature | Normalization |
|-------|---------|---------------|
| 3 | Minerals | ÷1000 |
| 4 | Vespene | ÷1000 |
| 5 | Supply used | ÷200 |
| 6 | Supply cap | ÷200 |

#### Supply State (2 features)
| Index | Feature | Type |
|-------|---------|------|
| 7 | Supply available | Binary |
| 8 | Supply full | Binary |

#### Spatial (8 features)
| Index | Feature | Notes |
|-------|---------|-------|
| 9-10 | Base position (nx, ny) | Normalized [0,1] |
| 11-12 | Army center (nx, ny) | Normalized [0,1] |
| 13-14 | Enemy center (nx, ny) | Normalized [0,1] |
| 15 | Army distance from base | Normalized 0-1 |
| 16 | Under attack | Binary |

#### Economy (5 features)
| Index | Feature | Normalization |
|-------|---------|---------------|
| 17 | CC count | ÷4 |
| 18 | SCV count | ÷70 |
| 19 | Refinery count | ÷6 |
| 20 | Idle workers | ÷10 |
| 21 | Production utilization | 0-1 |

#### Buildings (18 features)
Counts of all Terran structures, each normalized by expected maximum.

#### Army (18 features)
Counts of all Terran combat units, each normalized by expected maximum.

#### Army State (4 features)
| Index | Feature | Normalization |
|-------|---------|---------------|
| 56 | Army supply | ÷100 |
| 57 | Army health ratio | 0-1 |
| 58 | Units in combat | ÷30 |
| 59 | Army value ratio | ÷2 |

#### Enemy Info (10 features)
| Index | Feature | Normalization |
|-------|---------|---------------|
| 60 | Detected race | ÷3 |
| 61 | Enemy workers | ÷50 |
| 62 | Enemy bases | ÷4 |
| 63 | Enemy production | ÷8 |
| 64 | Enemy army supply | ÷100 |
| 65 | Enemy has air | Binary |
| 66-68 | Specific unit counts | Various |
| 69 | Enemy buildings | ÷10 |

#### Upgrades (6 features)
Binary flags for key researched upgrades (Stimpack, Combat Shield, etc.).

---

## Action System

### Action Registry (`terran_actions.py`)

The action registry maps action names to execution functions:

```python
ACTION_REGISTRY = {
    "do_nothing": do_nothing,
    "train_marine": train_marine,
    "build_barracks": build_barracks,
    ...
}
```

### Action Categories

#### Building Actions (16 actions)
- Core: supply_depot, barracks, factory, starport, command_center
- Tech: engineering_bay, armory, fusion_core, ghost_academy
- Defense: bunker, missile_turret, sensor_tower

#### Addon Actions (8 actions)
- barracks_techlab/reactor
- factory_techlab/reactor
- starport_techlab/reactor

#### Unit Training (17 actions)
- Barracks: marine, marauder, reaper, ghost
- Factory: hellion, hellbat, siege_tank, thor, cyclone, widow_mine
- Starport: viking, medivac, liberator, raven, banshee, battlecruiser

#### Abilities (2 actions)
- call_down_mule
- scanner_sweep

#### Combat Macros (6 actions)
- attack_aggressor: Bio/mech frontline assault
- attack_siege: Positioning siege units
- attack_support: Medivac/Raven positioning
- attack_harass: Fast unit harassment
- retreat_to_base: Defensive retreat
- final_push: All-in attack

### Legal Action Gating (`terran_legal_actions_logic.py`)

Each action has a precondition function that checks:
- Resource requirements (minerals, gas)
- Building/tech prerequisites
- Unit availability
- Supply requirements

```python
def pre_train_marine(obs):
    return (
        exists(obs, units.Terran.Barracks) and
        supply_available(obs) and
        minerals(obs) >= 50
    )
```

### Coordinate Requirements

Some actions require coordinates (building placement), tracked in `ACTION_NEEDS_COORDS`:

```python
ACTION_NEEDS_COORDS = {
    "build_supply_depot": True,
    "build_barracks": True,
    "train_marine": False,
    "attack_aggressor": False,
    ...
}
```

---

## Coordinate Transformation

### Problem

StarCraft II maps have two possible spawn positions (top-left, bottom-right). Without transformation, the network would need to learn separate strategies for each spawn.

### Solution (`coordinate_transform.py`)

```python
class CoordinateTransform:
    def __init__(self, map_width, map_height, cc_x, cc_y):
        # Detect spawn position
        self.needs_rotation = cc_x >= map_width/2 and cc_y >= map_height/2
```

### Transformation Pipeline

```
World Coordinates         Normalized Coordinates
(Absolute pixels)    →    [0, 1] range
     │                         │
     │  If bottom-right        │
     │  spawn: rotate 180°     │
     │                         │
     ▼                         ▼
┌─────────────────────────────────────┐
│ Top-left spawn: direct mapping      │
│ Bottom-right: rotated mapping       │
│                                     │
│ Result: CC always at ~(0, 0)        │
│         Enemy always in +direction  │
└─────────────────────────────────────┘
```

### Key Methods

```python
def world_to_normalized(self, x, y):
    """Convert world coords to [0,1] range"""
    if self.needs_rotation:
        x, y = self._rotate_180(x, y)
    nx = x / (self.map_width - 1)
    ny = y / (self.map_height - 1)
    return nx, ny

def normalized_to_world(self, nx, ny):
    """Convert normalized coords back to world"""
    x = nx * (self.map_width - 1)
    y = ny * (self.map_height - 1)
    if self.needs_rotation:
        x, y = self._rotate_180(x, y)
    return int(x), int(y)
```

---

## Distributed Training

### Architecture (`parallel_main_memory.py`, `gpu_server_memory.py`)

```
┌────────────────────────────────────────────────────────┐
│                    Main Process                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │              Queue Management                    │   │
│  │  prediction_queue ──► GPU Server ──► result_queue│   │
│  │  update_queue ──────► GPU Server                 │   │
│  │  control_queue ─────► GPU Server                 │   │
│  └─────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────┘
                            │
            ┌───────────────┼───────────────┐
            │               │               │
            ▼               ▼               ▼
       ┌─────────┐    ┌─────────┐    ┌─────────┐
       │Worker 0 │    │Worker 1 │    │Worker N │
       │  SC2    │    │  SC2    │    │  SC2    │
       └─────────┘    └─────────┘    └─────────┘
```

### Worker Process (`worker_memory.py`)

Each worker:
1. Creates SC2 environment
2. Runs episodes:
   - Sends state → prediction_queue
   - Receives action ← result_queue
   - Executes action in SC2
   - Accumulates experiences
3. Sends completed episode → update_queue

### GPU Server (`gpu_server_memory.py`)

The GPU server:
1. Handles prediction requests (batched)
2. Maintains LSTM hidden states per worker
3. Stores episodes in replay buffer
4. Trains every 16 episodes via PPO

### PPO Training

```python
# Compute ratio
ratio = exp(log_prob_new - log_prob_old)

# Clipped objective
surr1 = ratio * advantages
surr2 = clamp(ratio, 1-ε, 1+ε) * advantages
policy_loss = -min(surr1, surr2)

# Total loss
loss = policy_loss + 0.5 * coord_loss - 0.01 * entropy
```

---

## Reward System

### Current: Tiered Curriculum Learning (`worker_rewards.py`)

The reward system uses a 3-tier curriculum where each worker independently progresses through increasingly complex objectives:

#### Tier 1: Economy & Base Building (0-1.0 reward range)

**Instantaneous milestone rewards** (awarded once at achievement):

| Milestone | Threshold | Reward |
|-----------|-----------|--------|
| Workers | 16, 22, 30, 44, 55, 66 SCVs | 0.02-0.05 |
| Supply Depots | First 4 built | 0.02 each |
| Production Buildings | First barracks, factory, starport | 0.04-0.05 |
| Expansions | 2nd base, 3rd base | 0.06-0.08 |
| Base Completion | Fully functional bases | 0.10-0.12 |
| Addons | First techlab/reactor | 0.03 each |
| Tech Buildings | Engineering bay, armory | 0.02-0.03 |
| Orbital Command | First upgrade | 0.04 |
| Refineries | First 4 built | 0.02 each |

**Building placement bonus**: Up to +0.015 for buildings placed near Command Centers

#### Tier 2: Army Production & Tech (0-1.0 reward range)

| Milestone | Threshold | Reward |
|-----------|-----------|--------|
| Army Supply | 10, 20, 30, 50, 80, 100, 130 | 0.03-0.07 |
| Unit Diversity | 2, 4, 6, 8 unit types | 0.03-0.05 |
| Special Units | Siege Tank, Medivac, Thor, BC | 0.03-0.05 |
| Upgrades | 1, 2, 3, 4, 6 upgrades | 0.04-0.06 |
| Research Actions | Each unique research | 0.03 |
| Maintain Economy | Workers at 44, 55, 66 | 0.02-0.03 |

#### Tier 3: Combat Efficiency (normalized ±1 range)

- **Damage Dealt**: Enemy buildings destroyed (0.05 per building) + enemy army killed (0.03 per ~10 supply)
- **Damage Taken**: Tracked but not penalized step-by-step
- **Destruction Milestones**: 3, 6, 10+ buildings destroyed (0.05-0.08)
- **Attack Actions**: Strategic combat decisions (0.02 per action, max 5)
- **Win Bonus**: Time-based decay (1.0 for <3k steps, 0.1 for >9k steps)

### Graduation System

Workers advance to the next tier when **both** conditions are met:

```python
self.min_episodes_per_tier = 25    # Minimum episodes required
self.graduation_threshold = 0.85   # Must achieve 85% average
self.graduation_window = 10        # Over last 10 episodes
```

**Fast-track graduation**: Workers demonstrating consistent excellence can advance in 25 episodes (vs previous 50)
**Higher standards**: Must maintain 85% average performance (vs previous 70%) over last 10 episodes

### Discounting

```python
def compute_discounted_rewards(step_rewards, final_reward, gamma=0.997):
    """
    For instantaneous rewards, we don't need heavy discounting.
    Each step gets its own instant reward + share of terminal.
    """
    # Step rewards are already instantaneous - don't compound them
    # Distribute terminal reward to last 100 steps with decay
```

### Design Philosophy

1. **Instantaneous Rewards**: Each milestone rewarded once at achievement, not continuously
2. **Decoupling Tiers**: Each tier has independent action preferences via exploration weights
3. **Building Placement Optimization**: Rewards spatial coordination (closeness to CC)
4. **No Step Penalties**: Only rewards, never negative step rewards
5. **Merit-Based Progression**: Workers can fast-track through tiers by demonstrating excellence

---

## Base Building System

### BaseTracker (`base_building_system.py`)

Tracks building construction and gates expansion:

```python
# Building tiers
TIER_1_ANCHOR = {CommandCenter, OrbitalCommand, PlanetaryFortress}
TIER_2_PRODUCTION = {Barracks, Factory, Starport}
TIER_3_ACCESSORY = {SupplyDepot, EngineeringBay, Armory, ...}
```

### Expansion Logic

```
┌─────────────────────────────────────────┐
│           can_expand() Logic            │
├─────────────────────────────────────────┤
│                                         │
│  Current Base Complete?                 │
│  ├─ Anchor ≥ 1?                        │
│  ├─ Production ≥ 2?                    │
│  └─ Accessory ≥ 3?                     │
│                                         │
│  If all YES → Return available spots    │
│  If any NO  → Return empty list         │
│                                         │
└─────────────────────────────────────────┘
```

### Mineral Clustering

```python
def _cluster_minerals(self, minerals, cluster_distance=16):
    """Group minerals into base clusters"""
    # Uses greedy clustering algorithm
    # Returns list of mineral clusters
```

### Building Radius Enforcement

All buildings must be placed within radius 16 of a Command Center:

```python
def is_near_any_command_center(obs, x, y, max_radius=16):
    """HARD RESTRICTION: Check if position is near any CC"""
```

---

## Metrics and Visualization

### MetricsLogger (`metrics_logger.py`)

Logs per-episode:
- Actions taken with coordinates
- Building placements
- Economy snapshots
- Final unit/building counts
- Game outcome

### TrainingAnalyzer (`visualize_training.py`)

Generates dashboard with:
- Rolling win rate
- Game length distributions
- Unit composition analysis
- Building placement heatmaps
- Action distributions
- Win rate by spawn corner

---

## Performance Considerations

### Memory Management

- Episode memory buffer: 10,000 steps max
- Replay buffer: 64 episodes rolling window
- LSTM hidden states: Per-worker, cleared on episode reset

### GPU Utilization

- Batch predictions: Up to 64 requests per batch
- Training: 25 updates per training step
- Gradient clipping: max_norm=0.5

### SC2 Instance Management

- Staggered worker starts (3s delay)
- Graceful shutdown via control queue
- Process termination on Ctrl+C