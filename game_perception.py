"""
Comprehensive Game Perception for StarCraft II

Provides everything a human player would see:
- Own units/buildings/resources
- Enemy units/buildings (visible ones)
- Spatial information (army positions, base locations)
- Game time and phase
- Combat state
- Production status
- Upgrade status

STATE SIZE: 78 features
"""

from pysc2.lib import features, units
import numpy as np


class GamePerception:
    """Full game state perception for the LSTM."""
    
    COSTS = {
        "do_nothing": 0,
        "send_idle_workers_to_mine": 0,
        "send_idle_workers_to_gas": 0,
        "train_worker": 50,
        "build_supply_depot": 100,
        "build_command_center": 400,
        "build_barracks": 150,
        "build_factory": 150 + 100,
        "build_starport": 150 + 100,
        "build_refinery": 75,
        "build_engineering_bay": 125,
        "build_armory": 150 + 100,
        "build_fusion_core": 150 + 150,
        "build_ghost_academy": 150 + 50,
        "build_bunker": 100,
        "build_missile_turret": 100,
        "build_sensor_tower": 125 + 100,
        "add_barracks_techlab": 50 + 25,
        "add_barracks_reactor": 50 + 50,
        "add_factory_techlab": 50 + 25,
        "add_factory_reactor": 50 + 50,
        "add_starport_techlab": 50 + 25,
        "add_starport_reactor": 50 + 50,
        "build_techlab": 50 + 25,
        "build_reactor": 50 + 50,
        "train_marine": 50,
        "train_marauder": 100 + 25,
        "train_reaper": 50 + 50,
        "train_ghost": 150 + 125,
        "train_hellion": 100,
        "train_hellbat": 100,
        "train_siege_tank": 150 + 125,
        "train_thor": 300 + 200,
        "train_cyclone": 150 + 100,
        "train_widow_mine": 75 + 25,
        "train_viking": 125 + 75,
        "train_medivac": 100 + 100,
        "train_liberator": 150 + 125,
        "train_raven": 100 + 150,
        "train_banshee": 150 + 100,
        "train_battlecruiser": 400 + 300,
        "transform_hellion_to_hellbat": 0,
        "transform_hellbat_to_hellion": 0,
        "siege_tank_siege_mode": 0,
        "siege_tank_unsiege": 0,
        "viking_assault_mode": 0,
        "viking_fighter_mode": 0,
        "burrow_widow_mine": 0,
        "unburrow_widow_mine": 0,
        "call_down_mule": 0,
        "scanner_sweep": 0,
        "upgrade_orbital_command": 150,
        "upgrade_planetary_fortress": 150,
        "lower_depot": 0,
        "research_barracks_techlab": 100 + 100,
        "research_factory_techlab": 100 + 100,
        "research_starport_techlab": 100 + 100,
        "research_engineering_bay": 100 + 100,
        "research_armory": 100 + 100,
        "research_ghost_academy": 150 + 150,
        "attack_move": 0,
    }
    
    PRODUCER = {
        "train_worker": units.Terran.CommandCenter,
        "build_supply_depot": units.Terran.SCV,
        "build_barracks": units.Terran.SCV,
        "build_command_center": units.Terran.SCV,
        "train_marine": units.Terran.Barracks,
        "raise_depot": units.Terran.SupplyDepotLowered,
        "lower_depot": units.Terran.SupplyDepot,
        "liftoff": [units.Terran.CommandCenter, units.Terran.Barracks],
        "land": [units.Terran.CommandCenterFlying, units.Terran.BarracksFlying],
        "attack_move": units.Terran.Marine,
        "defend": units.Terran.Marine,
        "retreat": units.Terran.Marine,
        "scout_with_scv": units.Terran.SCV,
        "salvage_bunker": units.Terran.Bunker,
        "do_nothing": None,
        "build_factory": units.Terran.SCV,
        "build_starport": units.Terran.SCV,
        "build_fusion_core": units.Terran.SCV,
        "build_armory": units.Terran.SCV,
        "train_hellion": units.Terran.Factory,
        "train_siege_tank": units.Terran.Factory,
        "train_thor": units.Terran.Factory,
        "train_cyclone": units.Terran.Factory,
        "train_viking": units.Terran.Starport,
        "train_medivac": units.Terran.Starport,
        "train_liberator": units.Terran.Starport,
        "train_raven": units.Terran.Starport,
        "train_banshee": units.Terran.Starport,
        "train_battlecruiser": units.Terran.Starport,
        "build_sensor_tower": units.Terran.SCV,
        "build_ghost_academy": units.Terran.SCV,
        "build_missile_turret": units.Terran.SCV,
        "upgrade_orbital_command": units.Terran.CommandCenter,
        "upgrade_planetary_fortress": units.Terran.CommandCenter,
        "build_engineering_bay": units.Terran.SCV,
        "build_bunker": units.Terran.SCV,
        "train_ghost": units.Terran.GhostAcademy,
        "train_hellbat": units.Terran.Factory,
        "train_widow_mine": units.Terran.Factory,
        "call_down_mule": units.Terran.OrbitalCommand,
        "summon_auto_turret": units.Terran.Raven,
        "transform_hellion_to_hellbat": units.Terran.Hellion,
        "transform_hellbat_to_hellion": units.Terran.Hellbat,
        "research_stimpack": units.Terran.BarracksTechLab,
        "research_combat_shield": units.Terran.BarracksTechLab,
        "research_concussive_shells": units.Terran.BarracksTechLab,
        "research_siege_tech": units.Terran.FactoryTechLab,
        "research_infantry_weapons_1": units.Terran.EngineeringBay,
        "research_infantry_armor_1": units.Terran.EngineeringBay,
        "scanner_sweep": units.Terran.OrbitalCommand,
        "siege_tank_siege_mode": units.Terran.SiegeTank,
        "siege_tank_unsiege": units.Terran.SiegeTankSieged,
        "viking_assault_mode": units.Terran.VikingFighter,
        "viking_fighter_mode": units.Terran.VikingAssault,
        "burrow_widow_mine": units.Terran.WidowMine,
        "unburrow_widow_mine": units.Terran.WidowMineBurrowed,
    }
    
    BUILDING_TYPES = {
        # Terran
        units.Terran.CommandCenter, units.Terran.OrbitalCommand, units.Terran.PlanetaryFortress,
        units.Terran.SupplyDepot, units.Terran.Refinery, units.Terran.Barracks,
        units.Terran.EngineeringBay, units.Terran.MissileTurret, units.Terran.Bunker,
        units.Terran.SensorTower, units.Terran.Armory, units.Terran.Factory,
        units.Terran.GhostAcademy, units.Terran.Starport, units.Terran.FusionCore,
        # Zerg
        units.Zerg.Hatchery, units.Zerg.Lair, units.Zerg.Hive,
        units.Zerg.Extractor, units.Zerg.SpawningPool, units.Zerg.EvolutionChamber,
        units.Zerg.RoachWarren, units.Zerg.BanelingNest, units.Zerg.HydraliskDen,
        units.Zerg.Spire, units.Zerg.UltraliskCavern, units.Zerg.InfestationPit,
        units.Zerg.NydusNetwork, units.Zerg.NydusCanal, units.Zerg.GreaterSpire,
        units.Zerg.SpineCrawler, units.Zerg.SporeCrawler,
        # Protoss
        units.Protoss.Nexus, units.Protoss.Pylon, units.Protoss.Assimilator,
        units.Protoss.Gateway, units.Protoss.Forge, units.Protoss.CyberneticsCore,
        units.Protoss.PhotonCannon, units.Protoss.ShieldBattery, units.Protoss.TwilightCouncil,
        units.Protoss.RoboticsFacility, units.Protoss.RoboticsBay, units.Protoss.Stargate,
        units.Protoss.TemplarArchive, units.Protoss.DarkShrine, units.Protoss.FleetBeacon,
    }

    # Upgrade buff IDs (from SC2 data)
    BUFF_STIMPACK = 27
    BUFF_STIMPACK_MARAUDER = 28
    
    def __init__(self, obs):
        self.obs = obs
        self.player = obs.observation.player
        self.available_actions = set(getattr(obs.observation, 'available_actions', []))
        self.minerals = self.player.minerals
        self.vespene = self.player.vespene
        
        # Categorize all units
        self.units_self = []
        self.units_enemy = []
        self.units_neutral = []
        self.enemy_buildings = {}
        
        if hasattr(obs.observation, "raw_units") and len(obs.observation.raw_units) > 0:
            for u in obs.observation.raw_units:
                if u.alliance == features.PlayerRelative.SELF:
                    self.units_self.append(u)
                elif u.alliance == features.PlayerRelative.ENEMY:
                    self.units_enemy.append(u)
                    if u.unit_type in self.BUILDING_TYPES:
                        self.enemy_buildings[u.tag] = (u.x, u.y)
                elif u.alliance == features.PlayerRelative.NEUTRAL:
                    self.units_neutral.append(u)
            self.use_raw = True
        else:
            self.units_self = [
                u for u in getattr(obs.observation, "feature_units", [])
                if u.alliance == features.PlayerRelative.SELF
            ]
            self.use_raw = False
        
        # Cache game loop
        self._game_loop = obs.observation.game_loop[0] if hasattr(obs.observation, 'game_loop') else 0
    
    # ==========================================================================
    # GAME TIME & PHASE
    # ==========================================================================
    
    def game_loop(self):
        """Current game loop (16 loops = 1 game second at normal speed)"""
        return self._game_loop
    
    def game_seconds(self):
        """Approximate game time in seconds"""
        return self._game_loop / 22.4
    
    def game_minutes(self):
        """Game time in minutes"""
        return self.game_seconds() / 60.0
    
    def game_phase(self):
        """
        Returns game phase as float:
        0.0-0.33 = early game (0-5 min)
        0.33-0.66 = mid game (5-15 min)
        0.66-1.0 = late game (15+ min)
        """
        minutes = self.game_minutes()
        if minutes < 5:
            return minutes / 15.0
        elif minutes < 15:
            return 0.33 + (minutes - 5) / 30.0
        else:
            return min(1.0, 0.66 + (minutes - 15) / 45.0)
    
    # ==========================================================================
    # ENEMY INFORMATION
    # ==========================================================================
    
    def update_enemy_info(self, obs):
        """Update enemy building positions from current observation"""
        for u in obs.observation.raw_units:
            if u.alliance == features.PlayerRelative.ENEMY:
                if u.unit_type in self.BUILDING_TYPES:
                    self.enemy_buildings[u.tag] = (u.x, u.y)
    
    def enemy_unit_count(self, unit_type):
        """Count visible enemy units of a specific type"""
        return sum(1 for u in self.units_enemy if u.unit_type == unit_type)
    
    def enemy_army_supply(self):
        """Estimate enemy army supply from visible units"""
        supply = 0
        supply_costs = {
            # Terran
            units.Terran.Marine: 1, units.Terran.Marauder: 2, units.Terran.Reaper: 1,
            units.Terran.Ghost: 2, units.Terran.Hellion: 2, units.Terran.Hellbat: 2,
            units.Terran.SiegeTank: 3, units.Terran.SiegeTankSieged: 3,
            units.Terran.Cyclone: 3, units.Terran.Thor: 6,
            units.Terran.VikingFighter: 2, units.Terran.VikingAssault: 2,
            units.Terran.Medivac: 2, units.Terran.Liberator: 3, units.Terran.LiberatorAG: 3,
            units.Terran.Raven: 2, units.Terran.Banshee: 3, units.Terran.Battlecruiser: 6,
            units.Terran.WidowMine: 2, units.Terran.WidowMineBurrowed: 2,
            # Zerg
            units.Zerg.Zergling: 0.5, units.Zerg.Baneling: 0.5, units.Zerg.Roach: 2,
            units.Zerg.Ravager: 3, units.Zerg.Hydralisk: 2, units.Zerg.Lurker: 3,
            units.Zerg.Infestor: 2, units.Zerg.SwarmHost: 3, units.Zerg.Ultralisk: 6,
            units.Zerg.Mutalisk: 2, units.Zerg.Corruptor: 2, units.Zerg.BroodLord: 4,
            units.Zerg.Viper: 3, units.Zerg.Queen: 2,
            # Protoss
            units.Protoss.Zealot: 2, units.Protoss.Stalker: 2, units.Protoss.Sentry: 2,
            units.Protoss.Adept: 2, units.Protoss.HighTemplar: 2, units.Protoss.DarkTemplar: 2,
            units.Protoss.Archon: 4, units.Protoss.Immortal: 4, units.Protoss.Colossus: 6,
            units.Protoss.Disruptor: 3, units.Protoss.Observer: 1, units.Protoss.WarpPrism: 2,
            units.Protoss.Phoenix: 2, units.Protoss.VoidRay: 4, units.Protoss.Oracle: 3,
            units.Protoss.Carrier: 6, units.Protoss.Tempest: 5, units.Protoss.Mothership: 8,
        }
        for u in self.units_enemy:
            supply += supply_costs.get(u.unit_type, 0)
        return supply
    
    def enemy_worker_count(self):
        """Count visible enemy workers"""
        worker_types = {
            units.Terran.SCV, units.Terran.MULE,
            units.Zerg.Drone,
            units.Protoss.Probe
        }
        return sum(1 for u in self.units_enemy if u.unit_type in worker_types)
    
    def enemy_base_count(self):
        """Count visible enemy bases"""
        base_types = {
            units.Terran.CommandCenter, units.Terran.OrbitalCommand, units.Terran.PlanetaryFortress,
            units.Zerg.Hatchery, units.Zerg.Lair, units.Zerg.Hive,
            units.Protoss.Nexus
        }
        return sum(1 for u in self.units_enemy if u.unit_type in base_types)
    
    def enemy_production_count(self):
        """Count visible enemy production buildings"""
        prod_types = {
            units.Terran.Barracks, units.Terran.Factory, units.Terran.Starport,
            units.Zerg.SpawningPool, units.Zerg.RoachWarren, units.Zerg.HydraliskDen,
            units.Zerg.Spire, units.Zerg.UltraliskCavern,
            units.Protoss.Gateway, units.Protoss.WarpGate, units.Protoss.RoboticsFacility,
            units.Protoss.Stargate
        }
        return sum(1 for u in self.units_enemy if u.unit_type in prod_types)
    
    def enemy_has_air(self):
        """Check if enemy has visible air units"""
        air_types = {
            units.Terran.VikingFighter, units.Terran.Medivac,
            units.Terran.Liberator, units.Terran.LiberatorAG, units.Terran.Raven,
            units.Terran.Banshee, units.Terran.Battlecruiser,
            units.Zerg.Mutalisk, units.Zerg.Corruptor, units.Zerg.BroodLord,
            units.Zerg.Viper, units.Zerg.Overlord, units.Zerg.Overseer,
            units.Protoss.Phoenix, units.Protoss.VoidRay, units.Protoss.Oracle,
            units.Protoss.Carrier, units.Protoss.Tempest, units.Protoss.Mothership,
        }
        return any(u.unit_type in air_types for u in self.units_enemy)
    
    def detected_enemy_race(self):
        """
        Detect enemy race from visible units.
        Returns: 0=unknown, 1=terran, 2=zerg, 3=protoss
        """
        terran_types = {units.Terran.SCV, units.Terran.Marine, units.Terran.CommandCenter}
        zerg_types = {units.Zerg.Drone, units.Zerg.Zergling, units.Zerg.Hatchery, units.Zerg.Overlord}
        protoss_types = {units.Protoss.Probe, units.Protoss.Zealot, units.Protoss.Nexus, units.Protoss.Pylon}
        
        for u in self.units_enemy:
            if u.unit_type in terran_types:
                return 1
            elif u.unit_type in zerg_types:
                return 2
            elif u.unit_type in protoss_types:
                return 3
        return 0
    
    # ==========================================================================
    # SPATIAL AWARENESS
    # ==========================================================================
    
    def get_map_size(self):
        """Get map dimensions"""
        if hasattr(self.obs.observation, 'raw_data') and self.obs.observation.raw_data:
            raw = self.obs.observation.raw_data
            if hasattr(raw, 'map_state') and raw.map_state:
                size = raw.map_state.visibility.size
                return int(size.y), int(size.x)
        all_units = self.units_self + self.units_enemy + self.units_neutral
        if len(all_units) > 0:
            max_x = max(u.x for u in all_units)
            max_y = max(u.y for u in all_units)
            return int(max_y + 16), int(max_x + 16)
        return 88, 88
    
    def get_base_location(self):
        """Get main base location"""
        ccs = [u for u in self.units_self if u.unit_type in [
            units.Terran.CommandCenter, units.Terran.OrbitalCommand, 
            units.Terran.PlanetaryFortress
        ]]
        if ccs:
            return ccs[0].x, ccs[0].y
        h, w = self.get_map_size()
        return w // 2, h // 2
    
    def get_army_center(self):
        """Get center of mass of army units"""
        army = [u for u in self.units_self if u.unit_type not in {
            units.Terran.SCV, units.Terran.MULE,
            units.Terran.CommandCenter, units.Terran.OrbitalCommand, units.Terran.PlanetaryFortress,
            units.Terran.SupplyDepot, units.Terran.SupplyDepotLowered,
            units.Terran.Barracks, units.Terran.Factory, units.Terran.Starport,
            units.Terran.Refinery, units.Terran.EngineeringBay, units.Terran.Armory,
            units.Terran.FusionCore, units.Terran.GhostAcademy, units.Terran.Bunker,
            units.Terran.MissileTurret, units.Terran.SensorTower,
        }]
        if not army:
            return self.get_base_location()
        avg_x = sum(u.x for u in army) / len(army)
        avg_y = sum(u.y for u in army) / len(army)
        return avg_x, avg_y
    
    def get_enemy_army_center(self):
        """Get center of visible enemy army"""
        enemy_army = [u for u in self.units_enemy if u.unit_type not in self.BUILDING_TYPES]
        if not enemy_army:
            return None
        avg_x = sum(u.x for u in enemy_army) / len(enemy_army)
        avg_y = sum(u.y for u in enemy_army) / len(enemy_army)
        return avg_x, avg_y
    
    def army_distance_from_base(self):
        """How far is our army from our main base (normalized)"""
        base_x, base_y = self.get_base_location()
        army_x, army_y = self.get_army_center()
        h, w = self.get_map_size()
        max_dist = np.sqrt(w**2 + h**2)
        dist = np.sqrt((army_x - base_x)**2 + (army_y - base_y)**2)
        return dist / max_dist
    
    def get_start_corner(self):
        """Determine which corner we started in"""
        cc_loc = self.get_base_location()
        x, y = cc_loc
        h, w = self.get_map_size()
        center_x = w / 2
        center_y = h / 2
        if x < center_x and y < center_y:
            return "top_left"
        elif x >= center_x and y < center_y:
            return "top_right"
        elif x < center_x and y >= center_y:
            return "bottom_left"
        else:
            return "bottom_right"
    
    # ==========================================================================
    # COMBAT STATE
    # ==========================================================================
    
    def army_health_ratio(self):
        """Average health ratio of army units"""
        army = [u for u in self.units_self if u.unit_type not in {
            units.Terran.SCV, units.Terran.MULE
        } and u.unit_type not in self.BUILDING_TYPES]
        if not army:
            return 1.0
        # PySC2 raw units don't have health_max, just return 1.0 for now
        # or use u.health / 100.0 as rough approximation
        return 1.0
    
    def units_in_combat(self):
        """Estimate how many of our units are in combat (near enemies)"""
        combat_range = 10
        count = 0
        for u in self.units_self:
            if u.unit_type in self.BUILDING_TYPES:
                continue
            for e in self.units_enemy:
                dist = np.sqrt((u.x - e.x)**2 + (u.y - e.y)**2)
                if dist < combat_range:
                    count += 1
                    break
        return count
    
    def under_attack(self):
        """Check if our base is under attack"""
        base_x, base_y = self.get_base_location()
        attack_range = 20
        for e in self.units_enemy:
            if e.unit_type in self.BUILDING_TYPES:
                continue
            dist = np.sqrt((e.x - base_x)**2 + (e.y - base_y)**2)
            if dist < attack_range:
                return True
        return False
    
    # ==========================================================================
    # PRODUCTION & ECONOMY
    # ==========================================================================
    
    def idle_workers(self):
        """Count idle workers"""
        count = 0
        for u in self.units_self:
            if u.unit_type == units.Terran.SCV:
                if hasattr(u, 'order_length') and u.order_length == 0:
                    count += 1
        return count
    
    def production_buildings_busy(self):
        """Count production buildings with active orders"""
        prod_types = {
            units.Terran.Barracks, units.Terran.Factory, units.Terran.Starport,
            units.Terran.CommandCenter, units.Terran.OrbitalCommand
        }
        busy = 0
        total = 0
        for u in self.units_self:
            if u.unit_type in prod_types:
                total += 1
                if hasattr(u, 'order_length') and u.order_length > 0:
                    busy += 1
        return busy, total
    
    def income_rate(self):
        """Estimate income rate based on worker count and bases."""
        workers = self.scv_count()
        bases = self.cc_count()
        optimal_per_base = 16
        effective_workers = min(workers, bases * optimal_per_base + 6)
        return effective_workers * 40
    
    # ==========================================================================
    # ARMY VALUE
    # ==========================================================================
    
    def army_supply(self):
        """Calculate own army supply"""
        supply = 0
        supply_costs = {
            units.Terran.Marine: 1, units.Terran.Marauder: 2, units.Terran.Reaper: 1,
            units.Terran.Ghost: 2, units.Terran.Hellion: 2, units.Terran.Hellbat: 2,
            units.Terran.SiegeTank: 3, units.Terran.SiegeTankSieged: 3,
            units.Terran.Cyclone: 3, units.Terran.Thor: 6,
            units.Terran.VikingFighter: 2, units.Terran.VikingAssault: 2,
            units.Terran.Medivac: 2, units.Terran.Liberator: 3, units.Terran.LiberatorAG: 3,
            units.Terran.Raven: 2, units.Terran.Banshee: 3, units.Terran.Battlecruiser: 6,
            units.Terran.WidowMine: 2, units.Terran.WidowMineBurrowed: 2,
        }
        for u in self.units_self:
            supply += supply_costs.get(u.unit_type, 0)
        return supply
    
    def army_value_ratio(self):
        """Ratio of our army supply to visible enemy army supply"""
        ours = self.army_supply()
        theirs = self.enemy_army_supply()
        if theirs == 0:
            return 2.0 if ours > 0 else 1.0
        return min(2.0, ours / theirs)
    
    # ==========================================================================
    # UPGRADES
    # ==========================================================================
    
    def has_stimpack(self):
        """Check if stimpack is researched"""
        techlabs = [u for u in self.units_self if u.unit_type == units.Terran.BarracksTechLab]
        if not techlabs:
            return False
        for u in self.units_self:
            if u.unit_type in {units.Terran.Marine, units.Terran.Marauder}:
                if hasattr(u, 'buff_ids') and (self.BUFF_STIMPACK in u.buff_ids or self.BUFF_STIMPACK_MARAUDER in u.buff_ids):
                    return True
        return False
    
    def has_combat_shield(self):
        """Check if marines have combat shield"""
        for u in self.units_self:
            if u.unit_type == units.Terran.Marine:
                if hasattr(u, 'health_max') and u.health_max > 45:
                    return True
        return False
    
    def has_concussive_shells(self):
        """Check for concussive shells"""
        return self.barracks_techlab_count() > 0 and self.marauder_count() > 0
    
    def has_siege_tech(self):
        """Check if siege tech is researched"""
        return self.count(units.Terran.SiegeTankSieged) > 0
    
    def has_infantry_weapons_1(self):
        """Check for infantry weapons upgrade"""
        return self.engineering_bay_count() > 0 and self.game_minutes() > 8
    
    def has_infantry_armor_1(self):
        """Check for infantry armor upgrade"""
        return self.engineering_bay_count() > 0 and self.game_minutes() > 10
    
    # ==========================================================================
    # UNIT COUNTS
    # ==========================================================================
    
    def count(self, unit_type):
        return sum(1 for u in self.units_self if u.unit_type == unit_type)
    
    def exists(self, *types):
        return all(self.count(t) > 0 for t in types)
    
    def exists_any(self, *types):
        return any(self.count(t) > 0 for t in types)
    
    def unit_position(self, unit):
        return unit.x, unit.y
    
    # Building counts
    def barracks_count(self): 
        return self.count(units.Terran.Barracks)
    def barracks_flying_count(self): 
        return self.count(units.Terran.BarracksFlying)
    def barracks_techlab_count(self):
        return self.count(units.Terran.BarracksTechLab)
    def barracks_reactor_count(self):
        return self.count(units.Terran.BarracksReactor)
    def factory_count(self):
        return self.count(units.Terran.Factory)
    def factory_techlab_count(self):
        return self.count(units.Terran.FactoryTechLab)
    def factory_reactor_count(self):
        return self.count(units.Terran.FactoryReactor)
    def starport_count(self):
        return self.count(units.Terran.Starport)
    def starport_techlab_count(self):
        return self.count(units.Terran.StarportTechLab)
    def starport_reactor_count(self):
        return self.count(units.Terran.StarportReactor)
    def armory_count(self):
        return self.count(units.Terran.Armory) 
    def fusion_core_count(self):
        return self.count(units.Terran.FusionCore)
    def sensor_tower_count(self):
        return self.count(units.Terran.SensorTower)  
    def ghost_academy_count(self):
        return self.count(units.Terran.GhostAcademy) 
    def missile_turret_count(self):
        return self.count(units.Terran.MissileTurret) 
    def orbital_command_count(self):
        return self.count(units.Terran.OrbitalCommand)
    def planetary_fortress_count(self):
        return self.count(units.Terran.PlanetaryFortress)
    def engineering_bay_count(self):
        return self.count(units.Terran.EngineeringBay)
    def bunker_count(self):
        return self.count(units.Terran.Bunker)
    def refinery_count(self):
        return self.count(units.Terran.Refinery)
    def depot_count(self):
        return self.count(units.Terran.SupplyDepot) + self.count(units.Terran.SupplyDepotLowered)
    def cc_count(self):
        return (self.count(units.Terran.CommandCenter) + 
                self.count(units.Terran.CommandCenterFlying) +
                self.count(units.Terran.OrbitalCommand) +
                self.count(units.Terran.PlanetaryFortress))
    def techlab_count(self):
        return self.barracks_techlab_count() + self.factory_techlab_count() + self.starport_techlab_count()
    def reactor_count(self):
        return self.barracks_reactor_count() + self.factory_reactor_count() + self.starport_reactor_count()
    
    # Unit counts
    def scv_count(self): 
        return self.count(units.Terran.SCV)
    def marine_count(self): 
        return self.count(units.Terran.Marine)
    def marauder_count(self):
        return self.count(units.Terran.Marauder)
    def reaper_count(self):
        return self.count(units.Terran.Reaper)
    def ghost_count(self):
        return self.count(units.Terran.Ghost)
    def hellion_count(self):
        return self.count(units.Terran.Hellion)
    def hellbat_count(self):
        return self.count(units.Terran.Hellbat)
    def siege_tank_count(self):
        return self.count(units.Terran.SiegeTank) + self.count(units.Terran.SiegeTankSieged)
    def thor_count(self):
        return self.count(units.Terran.Thor)
    def cyclone_count(self):
        return self.count(units.Terran.Cyclone)
    def widow_mine_count(self):
        return self.count(units.Terran.WidowMine) + self.count(units.Terran.WidowMineBurrowed)
    def viking_count(self):
        return self.count(units.Terran.VikingFighter) + self.count(units.Terran.VikingAssault)
    def medivac_count(self):
        return self.count(units.Terran.Medivac)
    def liberator_count(self):
        return self.count(units.Terran.Liberator) + self.count(units.Terran.LiberatorAG)
    def raven_count(self):
        return self.count(units.Terran.Raven)
    def banshee_count(self):
        return self.count(units.Terran.Banshee)
    def battlecruiser_count(self):
        return self.count(units.Terran.Battlecruiser)
    def mule_count(self):
        return self.count(units.Terran.MULE)
    def auto_turret_count(self):
        return self.count(units.Terran.AutoTurret)
    
    # Supply
    @property
    def supply_used(self): 
        return self.player.food_used
    @property
    def supply_cap(self): 
        return self.player.food_cap
    def supply_available(self): 
        return self.supply_used < self.supply_cap
    def supply_full(self): 
        return self.supply_used >= self.supply_cap
    
    # Utility
    def has_minerals(self, action_name):
        return self.minerals >= self.COSTS.get(action_name, 0)
    def can_do(self, action_id):
        return action_id in self.available_actions
    def producer_unit_type(self, action_name):
        return self.PRODUCER.get(action_name)
    def producer_exists(self, action_name):
        producer = self.producer_unit_type(action_name)
        if producer is None:
            return True
        if isinstance(producer, list):
            return self.exists_any(*producer)
        return self.exists(producer)
    
    def did_win(self):
        if hasattr(self.obs, "player_result") and len(self.obs.player_result) > 0:
            return self.obs.player_result[0].result == 1
        return False
    
    def status(self):
        """Full status dict for debugging"""
        return {
            "minerals": self.minerals,
            "vespene": self.vespene,
            "supply": f"{self.supply_used}/{self.supply_cap}",
            "game_time": f"{self.game_minutes():.1f} min",
            "scvs": self.scv_count(),
            "army_supply": self.army_supply(),
            "enemy_army_supply": self.enemy_army_supply(),
            "army_value_ratio": f"{self.army_value_ratio():.2f}",
            "under_attack": self.under_attack(),
        }


def encode_state(perception, coord_transform):
    """
    Encode full game state for LSTM.
    
    All spatial coordinates are transformed through coord_transform to ensure
    consistent representation regardless of spawn position.
    
    Total features: 78
    
    Args:
        perception: GamePerception instance
        coord_transform: CoordinateTransform instance for normalizing coordinates
        
    Returns:
        numpy array of shape (78,) with float32 values
    """
    # Get raw world coordinates
    base_x, base_y = perception.get_base_location()
    army_x, army_y = perception.get_army_center()
    
    enemy_center = perception.get_enemy_army_center()
    if enemy_center:
        enemy_x, enemy_y = enemy_center
    else:
        # Default to opposite corner from base
        h, w = perception.get_map_size()
        enemy_x = w - base_x
        enemy_y = h - base_y
    
    # Transform all coordinates to normalized space [-1, 1]
    # After transform, our CC is always at (0, 0) and enemy is in positive direction
    base_nx, base_ny = coord_transform.world_to_normalized(base_x, base_y)
    army_nx, army_ny = coord_transform.world_to_normalized(army_x, army_y)
    enemy_nx, enemy_ny = coord_transform.world_to_normalized(enemy_x, enemy_y)
    
    busy, total = perception.production_buildings_busy()
    prod_utilization = busy / max(total, 1)

    state = np.array([
        # === GAME TIME (3) ===
        perception.game_minutes() / 30.0,
        perception.game_phase(),
        perception.game_loop() / 20000.0,
        
        # === RESOURCES (4) ===
        perception.minerals / 1000.0,
        perception.vespene / 1000.0,
        perception.supply_used / 200.0,
        perception.supply_cap / 200.0,
        
        # === SUPPLY STATE (2) ===
        1.0 if perception.supply_available() else 0.0,
        1.0 if perception.supply_full() else 0.0,
        
        # === SPATIAL (8) - NOW USES COORDINATE TRANSFORM ===
        base_nx,      # Will be ~0 since CC defines origin
        base_ny,      # Will be ~0 since CC defines origin  
        army_nx,      # Army position relative to CC [-1, 1]
        army_ny,
        enemy_nx,     # Enemy position relative to CC [-1, 1]
        enemy_ny,
        perception.army_distance_from_base(),  # Already normalized 0-1
        1.0 if perception.under_attack() else 0.0,
        
        # === ECONOMY (5) ===
        perception.cc_count() / 4.0,
        perception.scv_count() / 70.0,
        perception.refinery_count() / 6.0,
        perception.idle_workers() / 10.0,
        prod_utilization,
        
        # === OWN BUILDINGS (15) ===
        perception.depot_count() / 15.0,
        perception.barracks_count() / 8.0,
        perception.barracks_techlab_count() / 4.0,
        perception.barracks_reactor_count() / 4.0,
        perception.factory_count() / 4.0,
        perception.factory_techlab_count() / 2.0,
        perception.factory_reactor_count() / 2.0,
        perception.starport_count() / 4.0,
        perception.starport_techlab_count() / 2.0,
        perception.starport_reactor_count() / 2.0,
        perception.armory_count() / 2.0,
        perception.fusion_core_count() / 1.0,
        perception.engineering_bay_count() / 2.0,
        perception.ghost_academy_count() / 1.0,
        perception.bunker_count() / 4.0,
        
        # === DEFENSE BUILDINGS (3) ===
        perception.missile_turret_count() / 8.0,
        perception.sensor_tower_count() / 3.0,
        (perception.orbital_command_count() + perception.planetary_fortress_count()) / 4.0,
        
        # === OWN ARMY (18) ===
        perception.marine_count() / 50.0,
        perception.marauder_count() / 30.0,
        perception.reaper_count() / 10.0,
        perception.ghost_count() / 10.0,
        perception.hellion_count() / 20.0,
        perception.hellbat_count() / 20.0,
        perception.siege_tank_count() / 15.0,
        perception.thor_count() / 8.0,
        perception.cyclone_count() / 15.0,
        perception.widow_mine_count() / 15.0,
        perception.viking_count() / 20.0,
        perception.medivac_count() / 15.0,
        perception.liberator_count() / 15.0,
        perception.raven_count() / 5.0,
        perception.banshee_count() / 10.0,
        perception.battlecruiser_count() / 8.0,
        perception.mule_count() / 5.0,
        perception.auto_turret_count() / 5.0,
        
        # === ARMY STATE (4) ===
        perception.army_supply() / 100.0,
        perception.army_health_ratio(),
        perception.units_in_combat() / 30.0,
        perception.army_value_ratio() / 2.0,
        
        # === ENEMY INFO (10) ===
        perception.detected_enemy_race() / 3.0,
        perception.enemy_worker_count() / 50.0,
        perception.enemy_base_count() / 4.0,
        perception.enemy_production_count() / 8.0,
        perception.enemy_army_supply() / 100.0,
        1.0 if perception.enemy_has_air() else 0.0,
        perception.enemy_unit_count(units.Terran.Marine) / 30.0,
        perception.enemy_unit_count(units.Zerg.Zergling) / 40.0,
        perception.enemy_unit_count(units.Protoss.Zealot) / 20.0,
        len(perception.enemy_buildings) / 10.0,
        
        # === UPGRADES (6) ===
        1.0 if perception.has_stimpack() else 0.0,
        1.0 if perception.has_combat_shield() else 0.0,
        1.0 if perception.has_concussive_shells() else 0.0,
        1.0 if perception.has_siege_tech() else 0.0,
        1.0 if perception.has_infantry_weapons_1() else 0.0,
        1.0 if perception.has_infantry_armor_1() else 0.0,
        
    ], dtype=np.float32)
    
    return state


def extract_features(obs):
    """For backwards compatibility"""
    perception = GamePerception(obs)
    return {
        'minerals': perception.player.minerals,
        'vespene': perception.player.vespene,
        'supply_used': perception.supply_used,
        'supply_cap': perception.supply_cap,
        'scv_count': perception.scv_count(),
        'marine_count': perception.marine_count(),
        'game_time': perception.game_minutes(),
    }