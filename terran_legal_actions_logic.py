from pysc2.lib import features, units
from game_perception import GamePerception

def count_units(obs, unit_type):
    if not hasattr(obs.observation, "feature_units"):
        return 0
    return sum(
        1 for u in obs.observation.feature_units
        if u.unit_type == unit_type and u.alliance == features.PlayerRelative.SELF
    )
def exists(obs, unit_type):
    return count_units(obs, unit_type) > 0
def exists_any(obs, *unit_types):
    return any(count_units(obs, ut) > 0 for ut in unit_types)
def supply_available(obs):
    p = obs.observation.player
    return p.food_used < p.food_cap
def minerals(obs):
    return obs.observation.player.minerals
def vespene(obs):
    return obs.observation.player.vespene
MIN_COST_SCV = 50
MIN_COST_SUPPLY_DEPOT = 100
MIN_COST_BARRACKS = 150
MIN_COST_COMMAND_CENTER = 400
MIN_COST_MARINE = 50
MIN_COST_MARAUDER_MINERALS = 100
MIN_COST_MARAUDER_GAS = 25
MIN_COST_REAPER_MINERALS = 50
MIN_COST_REAPER_GAS = 50
MIN_COST_REFINERY = 75
MIN_COST_TECHLAB = 50
MIN_COST_REACTOR = 50
MIN_COST_FACTORY_MINERALS = 150
MIN_COST_FACTORY_GAS = 100
MIN_COST_ARMORY_MINERALS = 150
MIN_COST_ARMORY_GAS = 100
MIN_COST_HELLION = 100
MIN_COST_SIEGE_TANK_MINERALS = 150
MIN_COST_SIEGE_TANK_GAS = 125
MIN_COST_THOR_MINERALS = 300
MIN_COST_THOR_GAS = 200
MIN_COST_CYCLONE_MINERALS = 150
MIN_COST_CYCLONE_GAS = 100
MIN_COST_STARPORT_MINERALS = 150
MIN_COST_STARPORT_GAS = 100
MIN_COST_FUSION_CORE_MINERALS = 150
MIN_COST_FUSION_CORE_GAS = 150
MIN_COST_VIKING_MINERALS = 125
MIN_COST_VIKING_GAS = 75
MIN_COST_MEDIVAC_MINERALS = 100
MIN_COST_MEDIVAC_GAS = 100
MIN_COST_LIBERATOR_MINERALS = 150
MIN_COST_LIBERATOR_GAS = 125
MIN_COST_RAVEN_MINERALS = 100
MIN_COST_RAVEN_GAS = 150
MIN_COST_BANSHEE_MINERALS = 150
MIN_COST_BANSHEE_GAS = 100
MIN_COST_BATTLECRUISER_MINERALS = 400
MIN_COST_BATTLECRUISER_GAS = 300

def pre_do_nothing(obs):
    return True
def pre_train_worker(obs):
    return (
        exists(obs, units.Terran.CommandCenter)
        and supply_available(obs)
        and minerals(obs) >= MIN_COST_SCV
    )
def pre_build_refinery(obs):
    if minerals(obs) < MIN_COST_REFINERY or not exists(obs, units.Terran.SCV):
        return False
    geysers = [
        u for u in getattr(obs.observation, 'feature_units', [])
        if u.alliance == features.PlayerRelative.NEUTRAL and
           u.unit_type in (units.Neutral.VespeneGeyser, units.Neutral.RichVespeneGeyser)
    ]
    return len(geysers) > 0
def pre_send_idle_workers_to_gas(obs):
    refineries = [
        u for u in getattr(obs.observation, 'feature_units', [])
        if u.alliance == features.PlayerRelative.SELF and u.unit_type == units.Terran.Refinery
    ]
    if not refineries:
        return False
    idle_scvs = [
        u for u in getattr(obs.observation, 'feature_units', [])
        if u.unit_type == units.Terran.SCV and u.alliance == features.PlayerRelative.SELF
           and getattr(u, 'order_length', 1) == 0
    ]
    return len(idle_scvs) > 0
def pre_send_idle_workers_to_mine(obs):
    idle_scvs = [
        u for u in getattr(obs.observation, 'feature_units', [])
        if u.unit_type == units.Terran.SCV and u.alliance == features.PlayerRelative.SELF
           and getattr(u, 'order_length', 1) == 0
    ]
    return len(idle_scvs) > 0
def pre_build_supply_depot(obs):
    return exists(obs, units.Terran.SCV) and minerals(obs) >= MIN_COST_SUPPLY_DEPOT
def pre_build_barracks(obs):
    return (
        exists(obs, units.Terran.SCV)
        and exists_any(obs, units.Terran.SupplyDepot, units.Terran.SupplyDepotLowered)
        and minerals(obs) >= MIN_COST_BARRACKS
    )
def pre_build_factory(obs):
    p = obs.observation.player
    return (
        exists(obs, units.Terran.SCV)
        and exists_any(obs, units.Terran.SupplyDepot, units.Terran.SupplyDepotLowered)
        and p.minerals >= MIN_COST_FACTORY_MINERALS
        and p.vespene >= MIN_COST_FACTORY_GAS
    )

def pre_add_barracks_techlab(obs):
    barracks = [
        u for u in getattr(obs.observation, 'feature_units', [])
        if u.unit_type == units.Terran.Barracks and u.alliance == features.PlayerRelative.SELF
           and getattr(u, "add_on_tag", 0) == 0
    ]
    return len(barracks) > 0 and minerals(obs) >= MIN_COST_TECHLAB and vespene(obs) >= 25

def pre_add_barracks_reactor(obs):
    barracks = [
        u for u in getattr(obs.observation, 'feature_units', [])
        if u.unit_type == units.Terran.Barracks and u.alliance == features.PlayerRelative.SELF
           and getattr(u, "add_on_tag", 0) == 0
    ]
    return len(barracks) > 0 and minerals(obs) >= MIN_COST_REACTOR and vespene(obs) >= 50

def pre_add_factory_techlab(obs):
    factories = [
        u for u in getattr(obs.observation, 'feature_units', [])
        if u.unit_type == units.Terran.Factory and u.alliance == features.PlayerRelative.SELF
           and getattr(u, "add_on_tag", 0) == 0
    ]
    return len(factories) > 0 and minerals(obs) >= MIN_COST_TECHLAB and vespene(obs) >= 25

def pre_build_techlab(obs):
    # This is for barracks techlab based on your ACTION_REGISTRY mapping
    return pre_add_factory_techlab(obs)

def pre_build_reactor(obs):
    # This is for factory reactor based on your ACTION_REGISTRY mapping
    factories = [
        u for u in getattr(obs.observation, 'feature_units', [])
        if u.unit_type == units.Terran.Factory and u.alliance == features.PlayerRelative.SELF
           and getattr(u, "add_on_tag", 0) == 0
    ]
    return len(factories) > 0 and minerals(obs) >= MIN_COST_REACTOR and vespene(obs) >= 50

def pre_build_armory(obs):
    return (
        exists(obs, units.Terran.SCV)
        and exists(obs, units.Terran.Factory)
        and minerals(obs) >= MIN_COST_ARMORY_MINERALS
        and vespene(obs) >= MIN_COST_ARMORY_GAS
    )

def pre_train_siege_tank(obs):
    return (
        exists(obs, units.Terran.Factory)
        and supply_available(obs)
        and minerals(obs) >= MIN_COST_SIEGE_TANK_MINERALS
        and vespene(obs) >= MIN_COST_SIEGE_TANK_GAS
    )

def pre_train_thor(obs):
    return (
        exists(obs, units.Terran.Factory)
        and supply_available(obs)
        and minerals(obs) >= MIN_COST_THOR_MINERALS
        and vespene(obs) >= MIN_COST_THOR_GAS
    )

def pre_train_cyclone(obs):
    return (
        exists(obs, units.Terran.Factory)
        and supply_available(obs)
        and minerals(obs) >= MIN_COST_CYCLONE_MINERALS
        and vespene(obs) >= MIN_COST_CYCLONE_GAS
    )

def pre_build_starport(obs):
    return (
        exists(obs, units.Terran.SCV)
        and exists(obs, units.Terran.Factory)
        and minerals(obs) >= MIN_COST_STARPORT_MINERALS
        and vespene(obs) >= MIN_COST_STARPORT_GAS
    )

def pre_build_fusion_core(obs):
    return (
        exists(obs, units.Terran.SCV)
        and exists(obs, units.Terran.Starport)
        and minerals(obs) >= MIN_COST_FUSION_CORE_MINERALS
        and vespene(obs) >= MIN_COST_FUSION_CORE_GAS
    )

def pre_add_starport_techlab(obs):
    starports = [
        u for u in getattr(obs.observation, 'raw_units', [])
        if u.unit_type == units.Terran.Starport and u.alliance == features.PlayerRelative.SELF
           and u.build_progress == 100
           and getattr(u, "add_on_tag", 0) == 0
    ]
    return len(starports) > 0 and minerals(obs) >= MIN_COST_TECHLAB and vespene(obs) >= 25

def pre_add_starport_reactor(obs):
    starports = [
        u for u in getattr(obs.observation, 'raw_units', [])
        if u.unit_type == units.Terran.Starport and u.alliance == features.PlayerRelative.SELF
           and u.build_progress == 100
           and getattr(u, "add_on_tag", 0) == 0
    ]
    return len(starports) > 0 and minerals(obs) >= MIN_COST_REACTOR and vespene(obs) >= 50

def pre_train_viking(obs):
    return (
        exists(obs, units.Terran.Starport)
        and supply_available(obs)
        and minerals(obs) >= MIN_COST_VIKING_MINERALS
        and vespene(obs) >= MIN_COST_VIKING_GAS
    )

def pre_train_medivac(obs):
    return (
        exists(obs, units.Terran.Starport)
        and supply_available(obs)
        and minerals(obs) >= MIN_COST_MEDIVAC_MINERALS
        and vespene(obs) >= MIN_COST_MEDIVAC_GAS
    )

def pre_train_liberator(obs):
    return (
        exists(obs, units.Terran.Starport)
        and supply_available(obs)
        and minerals(obs) >= MIN_COST_LIBERATOR_MINERALS
        and vespene(obs) >= MIN_COST_LIBERATOR_GAS
    )

def pre_train_raven(obs):
    return (
        exists(obs, units.Terran.Starport)
        and supply_available(obs)
        and minerals(obs) >= MIN_COST_RAVEN_MINERALS
        and vespene(obs) >= MIN_COST_RAVEN_GAS
    )

def pre_train_banshee(obs):
    return (
        exists(obs, units.Terran.Starport)
        and supply_available(obs)
        and minerals(obs) >= MIN_COST_BANSHEE_MINERALS
        and vespene(obs) >= MIN_COST_BANSHEE_GAS
    )

def pre_train_battlecruiser(obs):
    return (
        exists(obs, units.Terran.Starport)
        and supply_available(obs)
        and minerals(obs) >= MIN_COST_BATTLECRUISER_MINERALS
        and vespene(obs) >= MIN_COST_BATTLECRUISER_GAS
    )

def pre_train_hellion(obs):
    return (
        exists(obs, units.Terran.Factory)
        and supply_available(obs)
        and minerals(obs) >= MIN_COST_HELLION
    )

def pre_train_marine(obs):
    return (
        exists(obs, units.Terran.Barracks)
        and supply_available(obs)
        and minerals(obs) >= MIN_COST_MARINE
    )
def pre_train_marauder(obs):
    return (
        exists(obs, units.Terran.Barracks)
        and supply_available(obs)
        and minerals(obs) >= MIN_COST_MARAUDER_MINERALS
        and vespene(obs) >= MIN_COST_MARAUDER_GAS
    )
def pre_train_reaper(obs):
    return (
        exists(obs, units.Terran.Barracks)
        and supply_available(obs)
        and minerals(obs) >= MIN_COST_REAPER_MINERALS
        and vespene(obs) >= MIN_COST_REAPER_GAS
    )
def pre_build_command_center(obs):
    return exists(obs, units.Terran.SCV) and minerals(obs) >= MIN_COST_COMMAND_CENTER
def pre_raise_depot(obs):
    return exists(obs, units.Terran.SupplyDepotLowered)
def pre_lower_depot(obs):
    return exists(obs, units.Terran.SupplyDepot)
def pre_liftoff(obs):
    return exists_any(obs, units.Terran.CommandCenter, units.Terran.Barracks)
def pre_land(obs):
    return exists_any(obs, units.Terran.CommandCenterFlying, units.Terran.BarracksFlying)
def pre_attack_move(obs):
    army_units = [
        u for u in getattr(obs.observation, "raw_units", [])
        if u.alliance == features.PlayerRelative.SELF and
           u.unit_type not in (
               units.Terran.SCV,
               units.Terran.MULE,
               units.Terran.CommandCenter,
               units.Terran.SupplyDepot,
               units.Terran.Barracks,
               units.Terran.BarracksFlying,
               units.Terran.BarracksTechLab,
               units.Terran.BarracksReactor,
               units.Terran.Refinery,
               units.Terran.Bunker,
               units.Terran.EngineeringBay,
               units.Terran.Factory,
               units.Terran.Starport,
               units.Terran.Armory,
           )
    ]
    return len(army_units) > 0
def pre_defend(obs):
    return pre_attack_move(obs)
def pre_retreat(obs):
    return pre_attack_move(obs)
def pre_scout_with_scv(obs):
    return exists(obs, units.Terran.SCV)
def pre_salvage_bunker(obs):
    return exists(obs, units.Terran.Bunker)

ACTION_REQUIREMENTS = {
    "do_nothing": pre_do_nothing,
    "send_idle_workers_to_mine": pre_send_idle_workers_to_mine,
    "train_worker": pre_train_worker,
    "build_supply_depot": pre_build_supply_depot,
    "build_barracks": pre_build_barracks,
    "build_command_center": pre_build_command_center,
    "train_marine": pre_train_marine,
    "train_marauder": pre_train_marauder,
    "train_reaper": pre_train_reaper,
    "lower_depot": pre_lower_depot,
    "attack_move": pre_attack_move,
    "build_refinery": pre_build_refinery,
    "send_idle_workers_to_gas": pre_send_idle_workers_to_gas,
    "add_barracks_techlab": pre_add_barracks_techlab,
    "add_barracks_reactor": pre_add_barracks_reactor,
    "build_techlab": pre_build_techlab,
    "build_reactor": pre_build_reactor,
    "build_factory": pre_build_factory,
    "train_hellion": pre_train_hellion,
    "add_factory_techlab": pre_add_factory_techlab,
    "build_armory": pre_build_armory,
    "train_siege_tank": pre_train_siege_tank,
    "train_thor": pre_train_thor,
    "train_cyclone": pre_train_cyclone,
    "build_starport": pre_build_starport,
    "build_fusion_core": pre_build_fusion_core,
    "add_starport_techlab": pre_add_starport_techlab,
    "add_starport_reactor": pre_add_starport_reactor,
    "train_viking": pre_train_viking,
    "train_medivac": pre_train_medivac,
    "train_liberator": pre_train_liberator,
    "train_raven": pre_train_raven,
    "train_banshee": pre_train_banshee,
    "train_battlecruiser": pre_train_battlecruiser,
}

def is_action_allowed(action_name, obs):
    func = ACTION_REQUIREMENTS.get(action_name)
    return func(obs) if func else False

def get_legal_actions(obs):
    return [name for name, rule in ACTION_REQUIREMENTS.items() if rule(obs)]