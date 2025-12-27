# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0] - 2025-01-XX

### Added
- Initial public release
- LSTM network with temporal attention (4-layer, 1024 hidden units, 8 attention heads)
- Map-agnostic coordinate transformation system
- Distributed training with 8 parallel workers
- Complete Terran tech tree (58 actions)
- 6 strategic combat macros (attack_aggressor, attack_siege, attack_support, attack_harass, retreat_to_base, final_push)
- PPO training with entropy bonus
- Base building system with expansion gating
- Comprehensive metrics logging
- Training visualization dashboard
- Pure win/loss reward system

### Architecture
- 78-feature state encoding covering game time, resources, spatial awareness, economy, buildings, army, enemy info, and upgrades
- Positional encoding for temporal understanding (up to 15,000 steps)
- Episode memory buffer for attention mechanism (10,000 steps)
- Sequence replay buffer for training (64 episodes)

### Training Infrastructure
- GPU server with batched predictions
- Worker processes with independent SC2 instances
- Checkpoint saving every 100 updates
- Graceful shutdown handling
- Crash logging per worker

### Removed
- Unit transformation actions (hellion↔hellbat, viking modes, etc.) - moved to combat macros
- Complex shaped rewards (base completion, unit production bonuses)
- Single attack_move action - replaced with 6 strategic combat macros

### Fixed
- Coordinate system bug with [-1,1] vs [0,1] normalization
- Double-discounting in reward calculation
- Terminal reward magnitude issues
- Memory leaks in visualization dashboard

## [0.9.0] - 2024-XX-XX (Internal)

### Added
- Curriculum-based training with 4 phases
- Base completion rewards (50/40/30/20 for main/natural/third/fourth)
- Single-worker optimization mode

### Changed
- Expanded action space from 61 to 67 actions
- Increased LSTM hidden size from 768 to 1024
- Adjusted memory buffer from 2,000 to 10,000 steps

### Issues Discovered
- Win rate plateaus at 30-40%
- Negative reward dominance
- Production cycle breakdown in late game

## [0.8.0] - 2024-XX-XX (Internal)

### Added
- Coordinate transformation for map-agnostic behavior
- Support for Simple64 and Simple128 maps
- Enemy building tracking across episodes

### Fixed
- Map size detection bug
- Spawn position detection
- Army center calculation

## [0.7.0] - 2024-XX-XX (Internal)

### Added
- Distributed training with parallel workers
- GPU server architecture
- Queue-based communication

### Changed
- Moved from single-process to multi-process training
- Added worker crash recovery

## [0.6.0] - 2024-XX-XX (Internal)

### Added
- Temporal attention mechanism
- Episode memory buffer
- Positional encoding

### Changed
- Upgraded from 2-layer to 3-layer LSTM
- Added layer normalization

## [0.5.0] - 2024-XX-XX (Internal)

### Added
- Basic LSTM network
- State encoding (initial 61 features)
- Action registry
- Legal action gating

### Infrastructure
- PySC2 integration
- Basic reward shaping
- Metrics logging

---

## Future Roadmap

### Planned for v1.1.0
- [ ] Multi-race support (Zerg, Protoss)
- [ ] Ladder map support
- [ ] Self-play training mode
- [ ] Model evaluation tools

### Planned for v1.2.0
- [ ] Transformer-based architecture option
- [ ] Imitation learning from replays
- [ ] Population-based training
- [ ] Docker containerization

### Under Consideration
- [ ] Web-based training dashboard
- [ ] Cloud training integration
- [ ] Battle.net ladder integration
- [ ] Multi-GPU distributed training