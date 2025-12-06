import json
import time
from collections import defaultdict
from pysc2.lib import features, units

class MetricsLogger:
    """
    Comprehensive metrics logging for StarCraft II bot training.
    Tracks everything that happened during a game for later analysis.
    """
    def __init__(self, log_file='training_metrics.jsonl'):
        self.log_file = log_file
        self.current_episode_data = None
        self.episode_number = 0
        
    def start_episode(self, episode_num):
        """Initialize data collection for a new episode"""
        self.episode_number = episode_num
        self.current_episode_data = {
            'episode': episode_num,
            'timestamp': time.time(),
            'actions_taken': [],
            'building_placements': [],
            'unit_production': defaultdict(int),
            'building_production': defaultdict(int),
            'economy_snapshots': [],
            'action_distribution': defaultdict(int),
        }
    
    def log_action(self, action_name, x_norm, y_norm, step, perception):
        """Log each action taken during the episode"""
        self.current_episode_data['actions_taken'].append({
            'step': step,
            'action': action_name,
            'x': float(x_norm),
            'y': float(y_norm),
            'minerals': perception.minerals,
            'vespene': perception.vespene,
            'supply': f"{perception.supply_used}/{perception.supply_cap}",
        })
        
        # Track action distribution
        self.current_episode_data['action_distribution'][action_name] += 1
        
        # Track building placements
        if action_name.startswith('build_'):
            self.current_episode_data['building_placements'].append({
                'building': action_name,
                'x': float(x_norm),
                'y': float(y_norm),
                'step': step,
            })
    
    def log_economy_snapshot(self, step, perception):
        """Take periodic snapshots of economy state"""
        self.current_episode_data['economy_snapshots'].append({
            'step': step,
            'minerals': perception.minerals,
            'vespene': perception.vespene,
            'supply_used': perception.supply_used,
            'supply_cap': perception.supply_cap,
            'scvs': perception.scv_count(),
            'total_army': (
                perception.marine_count() +
                perception.marauder_count() +
                perception.reaper_count() +
                perception.hellion_count() +
                perception.siege_tank_count() +
                perception.thor_count() +
                perception.cyclone_count() +
                perception.viking_count() +
                perception.medivac_count() +
                perception.liberator_count() +
                perception.raven_count() +
                perception.banshee_count() +
                perception.battlecruiser_count()
            )
        })
    
    def end_episode(self, outcome, total_steps, total_reward, perception, obs):
        """Finalize episode data and write to log"""
        # Final unit counts
        final_units = {
            'scvs': perception.scv_count(),
            'marines': perception.marine_count(),
            'marauders': perception.marauder_count(),
            'reapers': perception.reaper_count(),
            'hellions': perception.hellion_count(),
            'siege_tanks': perception.siege_tank_count(),
            'thors': perception.thor_count(),
            'cyclones': perception.cyclone_count(),
            'vikings': perception.viking_count(),
            'medivacs': perception.medivac_count(),
            'liberators': perception.liberator_count(),
            'ravens': perception.raven_count(),
            'banshees': perception.banshee_count(),
            'battlecruisers': perception.battlecruiser_count(),
        }
        
        # Final building counts
        final_buildings = {
            'command_centers': perception.cc_count(),
            'supply_depots': perception.depot_count(),
            'barracks': perception.barracks_count(),
            'barracks_techlabs': perception.barracks_techlab_count(),
            'barracks_reactors': perception.barracks_reactor_count(),
            'factories': perception.factory_count(),
            'factory_techlabs': perception.factory_techlab_count(),
            'factory_reactors': perception.factory_reactor_count(),
            'starports': perception.starport_count(),
            'starport_techlabs': perception.starport_techlab_count(),
            'starport_reactors': perception.starport_reactor_count(),
            'armories': perception.armory_count(),
            'fusion_cores': perception.fusion_core_count(),
        }
        
        # Calculate game time
        game_loops = obs.observation.game_loop
        game_time_seconds = game_loops / 22.4
        
        # Compile final episode data
        self.current_episode_data.update({
            'outcome': outcome,
            'total_steps': total_steps,
            'total_reward': float(total_reward),
            'game_time_seconds': float(game_time_seconds),
            'final_units': final_units,
            'final_buildings': final_buildings,
            'final_minerals': perception.minerals,
            'final_vespene': perception.vespene,
            'final_supply': f"{perception.supply_used}/{perception.supply_cap}",
        })
        
        # Calculate building placement heatmap data
        if self.current_episode_data['building_placements']:
            self.current_episode_data['placement_heatmap'] = self._calculate_heatmap(
                self.current_episode_data['building_placements']
            )
        
        # Write to log file
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(self.current_episode_data, default=int) + '\n')
    
    def _calculate_heatmap(self, placements):
        """Calculate heatmap bins for building placements"""
        # Create 10x10 grid
        heatmap = [[0 for _ in range(10)] for _ in range(10)]
        
        for p in placements:
            x_bin = min(9, int(p['x'] * 10))
            y_bin = min(9, int(p['y'] * 10))
            heatmap[y_bin][x_bin] += 1
        
        return heatmap
    
    def get_win_stats(self, last_n_episodes=None):
        """
        Analyze win statistics from the log file.
        Returns summary stats for won games.
        """
        wins = []
        
        try:
            with open(self.log_file, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    if data['outcome'] == 'win':
                        wins.append(data)
        except FileNotFoundError:
            return None
        
        if last_n_episodes:
            wins = wins[-last_n_episodes:]
        
        if not wins:
            return None
        
        # Calculate averages
        avg_game_time = sum(w['game_time_seconds'] for w in wins) / len(wins)
        avg_steps = sum(w['total_steps'] for w in wins) / len(wins)
        
        # Most common winning unit composition
        unit_totals = defaultdict(int)
        for w in wins:
            for unit, count in w['final_units'].items():
                unit_totals[unit] += count
        
        return {
            'total_wins': len(wins),
            'avg_game_time': avg_game_time,
            'avg_steps': avg_steps,
            'common_units': dict(sorted(unit_totals.items(), key=lambda x: x[1], reverse=True)[:5]),
            'latest_win': wins[-1] if wins else None
        }