"""
Metrics Logger - FIXED to use CoordinateTransform visualization space

All coordinates stored in visualization space [0, 1] where:
- CC is at (0.125, 0.125) for all games
- Coordinates are rotated for bottom-right spawns
- Ready for direct heatmap visualization
"""

import json
import time
from collections import defaultdict
from pysc2.lib import features, units

TERRAN_BUILDING_TYPES = {
    'build_command_center': units.Terran.CommandCenter,
    'build_orbital_command': units.Terran.OrbitalCommand,
    'build_planetary_fortress': units.Terran.PlanetaryFortress,
    'build_supply_depot': units.Terran.SupplyDepot,
    'build_refinery': units.Terran.Refinery,
    'build_barracks': units.Terran.Barracks,
    'build_engineering_bay': units.Terran.EngineeringBay,
    'build_missile_turret': units.Terran.MissileTurret,
    'build_bunker': units.Terran.Bunker,
    'build_sensor_tower': units.Terran.SensorTower,
    'build_armory': units.Terran.Armory,
    'build_factory': units.Terran.Factory,
    'build_ghost_academy': units.Terran.GhostAcademy,
    'build_starport': units.Terran.Starport,
    'build_fusion_core': units.Terran.FusionCore,
}

class MetricsLogger:
    EXCLUDED_ACTIONS = {
        'do_nothing',
        'attack_move',
        'move_camera',
        'select_army',
        'select_idle_worker',
    }
    
    BUILDING_ACTIONS = {
        'build_supply_depot',
        'build_barracks',
        'build_factory',
        'build_starport',
        'build_command_center',
        'build_engineering_bay',
        'build_armory',
        'build_ghost_academy',
        'build_fusion_core',
        'build_bunker',
        'build_missile_turret',
        'build_sensor_tower',
    }
    
    EXCLUDE_FROM_HEATMAP = {
        'build_refinery',
        'build_techlab_barracks',
        'build_reactor_barracks',
        'build_techlab_factory',
        'build_reactor_factory',
        'build_techlab_starport',
        'build_reactor_starport',
        'add_barracks_techlab',
        'add_barracks_reactor',
        'add_factory_techlab',
        'add_factory_reactor',
        'add_starport_techlab',
        'add_starport_reactor',
    }
    BUILDING_TIERS = {
        # Tier 1: Anchors (Command Centers)
        'build_command_center': 'anchor',
        'build_orbital_command': 'anchor',
        'build_planetary_fortress': 'anchor',
        
        # Tier 2: Production
        'build_barracks': 'production',
        'build_factory': 'production',
        'build_starport': 'production',
        
        # Tier 3: Accessory
        'build_supply_depot': 'accessory',
        'build_engineering_bay': 'accessory',
        'build_armory': 'accessory',
        'build_ghost_academy': 'accessory',
        'build_fusion_core': 'accessory',
        'build_bunker': 'accessory',
        'build_missile_turret': 'accessory',
        'build_sensor_tower': 'accessory',
    }
    def __init__(self, log_file='training_metrics.jsonl'):
        self.log_file = log_file
        self.current_episode_data = None
        self.episode_number = 0
        self.peak_units = defaultdict(int) 
        self.peak_buildings = defaultdict(int)
        self.previous_building_counts = defaultdict(int)
    
    def start_episode(self, episode_num, starting_corner=None):
        self.episode_number = episode_num
        self.peak_units = defaultdict(int)
        self.peak_buildings = defaultdict(int)
        self.previous_building_counts = defaultdict(int)
        
        self.current_episode_data = {
            'episode': episode_num,
            'timestamp': time.time(),
            'start_corner': starting_corner,
            'actions_taken': [],
            'building_placements': [],
            'unit_production': defaultdict(int),
            'building_production': defaultdict(int),
            'economy_snapshots': [],
            'action_distribution': defaultdict(int),
            'meaningful_actions': defaultdict(int),
        }
    
    def log_action(self, action_name, x_viz, y_viz, step, perception):
        """
        Log action with coordinates already in visualization space [0, 1].
        
        Args:
            action_name: Name of action
            x_viz, y_viz: Coordinates in visualization space where CC is at (0.125, 0.125)
            step: Current game step
            perception: GamePerception instance
        """
        x_vis = float(x_viz)
        y_vis = float(y_viz)
        
        # Log all actions
        self.current_episode_data['actions_taken'].append({
            'step': step,
            'action': action_name,
            'x': x_vis,
            'y': y_vis,
            'minerals': perception.minerals,
            'vespene': perception.vespene,
            'supply': f"{perception.supply_used}/{perception.supply_cap}",
        })
        
        self.current_episode_data['action_distribution'][action_name] += 1
        
        if action_name not in self.EXCLUDED_ACTIONS:
            self.current_episode_data['meaningful_actions'][action_name] += 1
        
        # Building placement validation
        if action_name in self.BUILDING_ACTIONS and action_name not in self.EXCLUDE_FROM_HEATMAP:
            # Check if valid coordinate (not on edges)
            is_valid_coord = (
                0.01 < x_vis < 0.99 and
                0.01 < y_vis < 0.99
            )
            
            if not is_valid_coord:
                return
            
            # Check if building count increased
            building_type = TERRAN_BUILDING_TYPES.get(action_name)
            if building_type:
                current_count = 0
                for u in perception.obs.observation.raw_units:
                    if u.alliance == features.PlayerRelative.SELF and u.unit_type == building_type:
                        current_count += 1
                
                if current_count > self.previous_building_counts[action_name]:
                    building_tier = self.BUILDING_TIERS.get(action_name, 'unknown')  # <-- ADD THIS LINE
                    self.current_episode_data['building_placements'].append({
                        'building': action_name,
                        'tier': building_tier,  # <-- NOW USE IT HERE
                        'x': x_vis,
                        'y': y_vis,
                        'step': step,
                    })
                    self.previous_building_counts[action_name] = current_count
    
    def log_economy_snapshot(self, step, perception):
        current_units = {
            'scvs': perception.scv_count(),
            'mules': perception.mule_count(),
            'marines': perception.marine_count(),
            'marauders': perception.marauder_count(),
            'reapers': perception.reaper_count(),
            'ghosts': perception.ghost_count(),
            'hellions': perception.hellion_count(),
            'hellbats': perception.hellbat_count(),
            'siege_tanks': perception.siege_tank_count(),
            'thors': perception.thor_count(),
            'cyclones': perception.cyclone_count(),
            'widow_mines': perception.widow_mine_count(),
            'vikings': perception.viking_count(),
            'medivacs': perception.medivac_count(),
            'liberators': perception.liberator_count(),
            'ravens': perception.raven_count(),
            'banshees': perception.banshee_count(),
            'battlecruisers': perception.battlecruiser_count(),
        }
        current_buildings = {
            'command_centers': perception.cc_count(),
            'orbital_commands': perception.orbital_command_count(),
            'planetary_fortresses': perception.planetary_fortress_count(),
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
            'engineering_bays': perception.engineering_bay_count(),
            'armories': perception.armory_count(),
            'fusion_cores': perception.fusion_core_count(),
            'ghost_academies': perception.ghost_academy_count(),
            'bunkers': perception.bunker_count(),
            'missile_turrets': perception.missile_turret_count(),
            'sensor_towers': perception.sensor_tower_count(),
        }
        
        for unit, count in current_units.items():
            if count > self.peak_units[unit]:
                self.peak_units[unit] = count
        
        for building, count in current_buildings.items():
            if count > self.peak_buildings[building]:
                self.peak_buildings[building] = count
        
        total_army = sum(current_units[u] for u in current_units if u not in ['scvs', 'mules'])
        
        self.current_episode_data['economy_snapshots'].append({
            'step': step,
            'minerals': perception.minerals,
            'vespene': perception.vespene,
            'supply_used': perception.supply_used,
            'supply_cap': perception.supply_cap,
            'scvs': current_units['scvs'],
            'total_army': total_army
        })
    
    def end_episode(self, outcome, total_steps, total_reward, perception, obs):
        final_units = defaultdict(int)
        final_buildings = defaultdict(int)
        
        for u in obs.observation.raw_units:
            if u.alliance == features.PlayerRelative.SELF:
                unit_type = u.unit_type
                if unit_type == units.Terran.SCV:
                    final_units['scvs'] += 1
                elif unit_type == units.Terran.MULE:
                    final_units['mules'] += 1
                elif unit_type == units.Terran.Marine:
                    final_units['marines'] += 1
                elif unit_type == units.Terran.Marauder:
                    final_units['marauders'] += 1
                elif unit_type == units.Terran.Reaper:
                    final_units['reapers'] += 1
                elif unit_type == units.Terran.Ghost:
                    final_units['ghosts'] += 1
                elif unit_type == units.Terran.Hellion:
                    final_units['hellions'] += 1
                elif unit_type == units.Terran.Hellbat:
                    final_units['hellbats'] += 1
                elif unit_type in [units.Terran.SiegeTank, units.Terran.SiegeTankSieged]:
                    final_units['siege_tanks'] += 1
                elif unit_type == units.Terran.Thor:
                    final_units['thors'] += 1
                elif unit_type == units.Terran.Cyclone:
                    final_units['cyclones'] += 1
                elif unit_type in [units.Terran.WidowMine, units.Terran.WidowMineBurrowed]:
                    final_units['widow_mines'] += 1
                elif unit_type in [units.Terran.VikingFighter, units.Terran.VikingAssault]:
                    final_units['vikings'] += 1
                elif unit_type == units.Terran.Medivac:
                    final_units['medivacs'] += 1
                elif unit_type in [units.Terran.Liberator, units.Terran.LiberatorAG]:
                    final_units['liberators'] += 1
                elif unit_type == units.Terran.Raven:
                    final_units['ravens'] += 1
                elif unit_type == units.Terran.Banshee:
                    final_units['banshees'] += 1
                elif unit_type == units.Terran.Battlecruiser:
                    final_units['battlecruisers'] += 1
                elif unit_type == units.Terran.CommandCenter:
                    final_buildings['command_centers'] += 1
                elif unit_type == units.Terran.OrbitalCommand:
                    final_buildings['orbital_commands'] += 1
                elif unit_type == units.Terran.PlanetaryFortress:
                    final_buildings['planetary_fortresses'] += 1
                elif unit_type in [units.Terran.SupplyDepot, units.Terran.SupplyDepotLowered]:
                    final_buildings['supply_depots'] += 1
                elif unit_type == units.Terran.Refinery:
                    final_buildings['refineries'] += 1
                elif unit_type == units.Terran.Barracks:
                    final_buildings['barracks'] += 1
                elif unit_type == units.Terran.BarracksTechLab:
                    final_buildings['barracks_techlabs'] += 1
                elif unit_type == units.Terran.BarracksReactor:
                    final_buildings['barracks_reactors'] += 1
                elif unit_type == units.Terran.Factory:
                    final_buildings['factories'] += 1
                elif unit_type == units.Terran.FactoryTechLab:
                    final_buildings['factory_techlabs'] += 1
                elif unit_type == units.Terran.FactoryReactor:
                    final_buildings['factory_reactors'] += 1
                elif unit_type == units.Terran.Starport:
                    final_buildings['starports'] += 1
                elif unit_type == units.Terran.StarportTechLab:
                    final_buildings['starport_techlabs'] += 1
                elif unit_type == units.Terran.StarportReactor:
                    final_buildings['starport_reactors'] += 1
                elif unit_type == units.Terran.EngineeringBay:
                    final_buildings['engineering_bays'] += 1
                elif unit_type == units.Terran.Armory:
                    final_buildings['armories'] += 1
                elif unit_type == units.Terran.FusionCore:
                    final_buildings['fusion_cores'] += 1
                elif unit_type == units.Terran.GhostAcademy:
                    final_buildings['ghost_academies'] += 1
                elif unit_type == units.Terran.Bunker:
                    final_buildings['bunkers'] += 1
                elif unit_type == units.Terran.MissileTurret:
                    final_buildings['missile_turrets'] += 1
                elif unit_type == units.Terran.SensorTower:
                    final_buildings['sensor_towers'] += 1
        
        game_loops = obs.observation.game_loop
        game_time_seconds = game_loops / 22.4
        
        self.current_episode_data.update({
            'outcome': outcome,
            'total_steps': total_steps,
            'total_reward': float(total_reward),
            'game_time_seconds': float(game_time_seconds),
            'final_units': dict(final_units),
            'final_buildings': dict(final_buildings),
            'peak_units': dict(self.peak_units),
            'peak_buildings': dict(self.peak_buildings),
            'final_minerals': perception.minerals,
            'final_vespene': perception.vespene,
            'final_supply': f"{perception.supply_used}/{perception.supply_cap}",
        })
        
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(self.current_episode_data, default=int) + '\n')
    
    def get_win_stats(self, last_n_episodes=None):
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
        
        avg_game_time = sum(w['game_time_seconds'] for w in wins) / len(wins)
        avg_steps = sum(w['total_steps'] for w in wins) / len(wins)

        unit_totals = defaultdict(int)
        for w in wins:
            units_data = w.get('peak_units', w.get('final_units', {}))
            for unit, count in units_data.items():
                unit_totals[unit] += count
        
        return {
            'total_wins': len(wins),
            'avg_game_time': avg_game_time,
            'avg_steps': avg_steps,
            'common_units': dict(sorted(unit_totals.items(), key=lambda x: x[1], reverse=True)[:5]),
            'latest_win': wins[-1] if wins else None
        }