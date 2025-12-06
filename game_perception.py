from pysc2.lib import features, units

class GamePerception:
    COSTS = {
        "train_worker": 50,
        "build_supply_depot": 100,
        "build_barracks": 150,
        "train_marine": 50,
        "build_command_center": 400,
        "train_marauder": 100 + 25, 
        "train_reaper": 50 + 50,
        "build_factory": 150 + 100,
        "build_starport": 150 + 100,
        "build_fusion_core": 150 + 150,
        "build_armory": 150 + 100,
        "build_techlab": 50 + 25,
        "build_reactor": 50 + 50,
        "train_hellion": 100,
        "train_siege_tank": 150 + 125,
        "train_thor": 300 + 200,
        "train_cyclone": 150 + 100,
        "train_viking": 125 + 75,
        "train_medivac": 100 + 100,
        "train_liberator": 150 + 125,
        "train_raven": 100 + 150,
        "train_banshee": 150 + 100,
        "train_battlecruiser": 400 + 300,
    }
    PRODUCER = {
        "train_worker": units.Terran.CommandCenter,
        "build_supply_depot": units.Terran.SCV,
        "build_barracks": units.Terran.SCV,
        "build_command_center": units.Terran.SCV,
        "train_marine": units.Terran.Barracks,
        "raise_depot": units.Terran.SupplyDepotLowered,
        "lower_depot": units.Terran.SupplyDepot,
        "liftoff": [
            units.Terran.CommandCenter,
            units.Terran.Barracks
        ],
        "land": [
            units.Terran.CommandCenterFlying,
            units.Terran.BarracksFlying
        ],
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
    }
    def __init__(self, obs):
        self.obs = obs
        self.player = obs.observation.player
        self.available_actions = set(getattr(obs.observation, 'available_actions', []))
        self.minerals = self.player.minerals
        self.vespene = self.player.vespene
        if hasattr(obs.observation, "raw_units") and len(obs.observation.raw_units) > 0:
            self.units_self = [
                u for u in obs.observation.raw_units
                if u.alliance == features.PlayerRelative.SELF
            ]
            self.use_raw = True
        else:
            self.units_self = [
                u for u in getattr(obs.observation, "feature_units", [])
                if u.alliance == features.PlayerRelative.SELF
            ]
            self.use_raw = False
    def unit_position(self, unit):
        if self.use_raw:
            return unit.x, unit.y
        return unit.x, unit.y
    def get_base_location(self):
        ccs = [u for u in self.units_self if u.unit_type == units.Terran.CommandCenter]
        if ccs:
            return self.unit_position(ccs[0])
        return (42, 42)
    def get_start_corner(self):
        cc_loc = self.get_base_location()
        x, y = cc_loc
        if x < 42 and y < 42:
            return "top_left"
        elif x >= 42 and y < 42:
            return "top_right"
        elif x < 42 and y >= 42:
            return "bottom_left"
        else:
            return "bottom_right"
    def count(self, unit_type):
        return sum(1 for u in self.units_self if u.unit_type == unit_type)
    def exists(self, *types):
        return all(self.count(t) > 0 for t in types)
    def exists_any(self, *types):
        return any(self.count(t) > 0 for t in types)
    
    # Barracks
    def barracks_count(self): 
        return self.count(units.Terran.Barracks)
    def barracks_flying_count(self): 
        return self.count(units.Terran.BarracksFlying)
    def barracks_techlab_count(self):
        return sum(1 for u in self.units_self
                if u.unit_type == units.Terran.Barracks 
                and getattr(u, "add_on_tag", 0) != 0
                and getattr(u, "add_on_type", None) == units.Terran.TechLab)
    def barracks_reactor_count(self):
        return sum(1 for u in self.units_self
                if u.unit_type == units.Terran.Barracks 
                and getattr(u, "add_on_tag", 0) != 0
                and getattr(u, "add_on_type", None) == units.Terran.Reactor)
    
    # Factory
    def factory_count(self):
        return self.count(units.Terran.Factory)
    def factory_techlab_count(self):
        return sum(1 for u in self.units_self
                if u.unit_type == units.Terran.Factory 
                and getattr(u, "add_on_tag", 0) != 0
                and getattr(u, "add_on_type", None) == units.Terran.TechLab)
    def factory_reactor_count(self):
        return sum(1 for u in self.units_self
                if u.unit_type == units.Terran.Factory 
                and getattr(u, "add_on_tag", 0) != 0
                and getattr(u, "add_on_type", None) == units.Terran.Reactor)
    
    # Starport
    def starport_count(self):
        return self.count(units.Terran.Starport)
    def starport_techlab_count(self):
        return sum(1 for u in self.units_self
                if u.unit_type == units.Terran.Starport 
                and getattr(u, "add_on_tag", 0) != 0 
                and getattr(u, "add_on_type", None) == units.Terran.TechLab)
    def starport_reactor_count(self):
        return sum(1 for u in self.units_self
                if u.unit_type == units.Terran.Starport 
                and getattr(u, "add_on_tag", 0) != 0 
                and getattr(u, "add_on_type", None) == units.Terran.Reactor)
    
    # Other structures
    def armory_count(self):
        return self.count(units.Terran.Armory)
    def fusion_core_count(self):
        return self.count(units.Terran.FusionCore)
    
    # Infantry units
    def scv_count(self): 
        return self.count(units.Terran.SCV)
    def marine_count(self): 
        return self.count(units.Terran.Marine)
    def marauder_count(self):
        return self.count(units.Terran.Marauder)
    def reaper_count(self):
        return self.count(units.Terran.Reaper)
    
    # Factory units
    def hellion_count(self):
        return self.count(units.Terran.Hellion)
    def siege_tank_count(self):
        return self.count(units.Terran.SiegeTank) + self.count(units.Terran.SiegeTankSieged)
    def thor_count(self):
        return self.count(units.Terran.Thor)
    def cyclone_count(self):
        return self.count(units.Terran.Cyclone)
    
    # Starport units
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
    
    # Legacy methods (for backwards compatibility with barracks)
    def depot_count(self):
        return self.count(units.Terran.SupplyDepot) + self.count(units.Terran.SupplyDepotLowered)
    def cc_count(self):
        return self.count(units.Terran.CommandCenter) + self.count(units.Terran.CommandCenterFlying)
    def techlab_count(self):
        # Legacy: returns barracks techlabs only
        return self.barracks_techlab_count()
    def reactor_count(self):
        # Legacy: returns barracks reactors only
        return self.barracks_reactor_count()
    
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
    def status(self):
        return {
            "minerals": self.minerals,
            "vespene": self.vespene,
            "supply": f"{self.supply_used}/{self.supply_cap}",
            "scvs": self.scv_count(),
            "marines": self.marine_count(),
            "marauders": self.marauder_count(),
            "reapers": self.reaper_count(),
            "hellions": self.hellion_count(),
            "siege_tanks": self.siege_tank_count(),
            "thors": self.thor_count(),
            "cyclones": self.cyclone_count(),
            "vikings": self.viking_count(),
            "medivacs": self.medivac_count(),
            "liberators": self.liberator_count(),
            "ravens": self.raven_count(),
            "banshees": self.banshee_count(),
            "battlecruisers": self.battlecruiser_count(),
            "depots": self.depot_count(),
            "barracks": self.barracks_count(),
            "barracks_techlabs": self.barracks_techlab_count(),
            "barracks_reactors": self.barracks_reactor_count(),
            "factories": self.factory_count(),
            "factory_techlabs": self.factory_techlab_count(),
            "factory_reactors": self.factory_reactor_count(),
            "starports": self.starport_count(),
            "starport_techlabs": self.starport_techlab_count(),
            "starport_reactors": self.starport_reactor_count(),
            "armory": self.armory_count(),
            "fusion_core": self.fusion_core_count(),
            "cc": self.cc_count(),
            "available_actions": list(self.available_actions),
        }
    def did_win(self):
        if hasattr(self.obs, "player_result") and len(self.obs.player_result) > 0:
            return self.obs.player_result[0].result == 1
        return False

def extract_features(obs):
    perception = GamePerception(obs)
    return {
        'minerals': perception.player.minerals,
        'vespene': perception.player.vespene,
        'supply_used': perception.supply_used,
        'supply_cap': perception.supply_cap,
        'scv_count': perception.scv_count(),
        'marine_count': perception.marine_count(),
        'marauder_count': perception.marauder_count(),
        'reaper_count': perception.reaper_count(),
        'hellion_count': perception.hellion_count(),
        'siege_tank_count': perception.siege_tank_count(),
        'thor_count': perception.thor_count(),
        'cyclone_count': perception.cyclone_count(),
        'viking_count': perception.viking_count(),
        'medivac_count': perception.medivac_count(),
        'liberator_count': perception.liberator_count(),
        'raven_count': perception.raven_count(),
        'banshee_count': perception.banshee_count(),
        'battlecruiser_count': perception.battlecruiser_count(),
        'depot_count': perception.depot_count(),
        'barracks_count': perception.barracks_count(),
        'barracks_techlab_count': perception.barracks_techlab_count(),
        'barracks_reactor_count': perception.barracks_reactor_count(),
        'factory_count': perception.factory_count(),
        'factory_techlab_count': perception.factory_techlab_count(),
        'factory_reactor_count': perception.factory_reactor_count(),
        'starport_count': perception.starport_count(),
        'starport_techlab_count': perception.starport_techlab_count(),
        'starport_reactor_count': perception.starport_reactor_count(),
        'armory_count': perception.armory_count(),
        'fusion_core_count': perception.fusion_core_count(),
        'cc_count': perception.cc_count(),
        'base_location': perception.get_base_location(),
        'start_corner': perception.get_start_corner(),
    }