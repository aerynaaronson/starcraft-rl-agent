from pysc2.lib import features, units
from game_perception import GamePerception

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
MIN_COST_STIMPACK_MINERALS = 100
MIN_COST_STIMPACK_GAS = 100
MIN_COST_COMBAT_SHIELD_MINERALS = 100
MIN_COST_COMBAT_SHIELD_GAS = 100
MIN_COST_CONCUSSIVE_SHELLS_MINERALS = 50
MIN_COST_CONCUSSIVE_SHELLS_GAS = 50
MIN_COST_SIEGE_TECH_MINERALS = 100
MIN_COST_SIEGE_TECH_GAS = 100
MIN_COST_INFANTRY_WEAPONS_MINERALS = 100
MIN_COST_INFANTRY_WEAPONS_GAS = 100
MIN_COST_INFANTRY_ARMOR_MINERALS = 100
MIN_COST_INFANTRY_ARMOR_GAS = 100
MIN_COST_SENSOR_TOWER_MINERALS = 125
MIN_COST_SENSOR_TOWER_GAS = 100
MIN_COST_GHOST_ACADEMY_MINERALS = 150
MIN_COST_GHOST_ACADEMY_GAS = 50
MIN_COST_MISSILE_TURRET = 100
MIN_COST_ORBITAL_COMMAND = 150
MIN_COST_PLANETARY_FORTRESS = 150
MIN_COST_ENGINEERING_BAY = 125
MIN_COST_BUNKER = 100
MIN_COST_GHOST_MINERALS = 150
MIN_COST_GHOST_GAS = 125
MIN_COST_WIDOW_MINE_MINERALS = 75
MIN_COST_WIDOW_MINE_GAS = 25

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
    return pre_add_factory_techlab(obs)
def pre_build_reactor(obs):
    factories = [
        u for u in getattr(obs.observation, 'feature_units', [])
        if u.unit_type == units.Terran.Factory and u.alliance == features.PlayerRelative.SELF
           and getattr(u, "add_on_tag", 0) == 0
    ]
    return len(factories) > 0 and minerals(obs) >= MIN_COST_REACTOR and vespene(obs) >= 50
def pre_research_stimpack(obs):
    barracks_techlabs = [u for u in getattr(obs.observation, 'raw_units', [])
                         if u.unit_type == units.Terran.BarracksTechLab
                         and u.alliance == features.PlayerRelative.SELF]
    return (
        len(barracks_techlabs) > 0
        and minerals(obs) >= MIN_COST_STIMPACK_MINERALS
        and vespene(obs) >= MIN_COST_STIMPACK_GAS
    )
def pre_research_combat_shield(obs):
    barracks_techlabs = [u for u in getattr(obs.observation, 'raw_units', [])
                         if u.unit_type == units.Terran.BarracksTechLab
                         and u.alliance == features.PlayerRelative.SELF]
    return (
        len(barracks_techlabs) > 0
        and minerals(obs) >= MIN_COST_COMBAT_SHIELD_MINERALS
        and vespene(obs) >= MIN_COST_COMBAT_SHIELD_GAS
    )
def pre_build_sensor_tower(obs):
    return (
        exists(obs, units.Terran.SCV)
        and exists(obs, units.Terran.EngineeringBay)
        and minerals(obs) >= MIN_COST_SENSOR_TOWER_MINERALS
        and vespene(obs) >= MIN_COST_SENSOR_TOWER_GAS
    )
def pre_build_ghost_academy(obs):
    return (
        exists(obs, units.Terran.SCV)
        and exists(obs, units.Terran.Barracks)
        and minerals(obs) >= MIN_COST_GHOST_ACADEMY_MINERALS
        and vespene(obs) >= MIN_COST_GHOST_ACADEMY_GAS
        and count_units(obs, units.Terran.GhostAcademy) < 1  # MAX 1
    )
def pre_build_missile_turret(obs):
    return (
        exists(obs, units.Terran.SCV)
        and exists(obs, units.Terran.EngineeringBay)
        and minerals(obs) >= MIN_COST_MISSILE_TURRET
    )
def pre_build_engineering_bay(obs):
    return (
        exists(obs, units.Terran.SCV)
        and exists_any(obs, units.Terran.SupplyDepot, units.Terran.SupplyDepotLowered)
        and minerals(obs) >= MIN_COST_ENGINEERING_BAY
    )
def pre_build_bunker(obs):
    return (
        exists(obs, units.Terran.SCV)
        and exists(obs, units.Terran.Barracks)
        and minerals(obs) >= MIN_COST_BUNKER
        and count_units(obs, units.Terran.Bunker) < 1  # MAX 1 BUNKER
    )
def pre_upgrade_orbital_command(obs):
    ccs = [u for u in getattr(obs.observation, 'feature_units', [])
           if u.unit_type == units.Terran.CommandCenter 
           and u.alliance == features.PlayerRelative.SELF
           and getattr(u, 'build_progress', 100) == 100]
    return len(ccs) > 0 and minerals(obs) >= MIN_COST_ORBITAL_COMMAND
def pre_upgrade_planetary_fortress(obs):
    ccs = [u for u in getattr(obs.observation, 'feature_units', [])
           if u.unit_type == units.Terran.CommandCenter 
           and u.alliance == features.PlayerRelative.SELF
           and getattr(u, 'build_progress', 100) == 100]
    return (
        len(ccs) > 0 
        and exists(obs, units.Terran.EngineeringBay)
        and minerals(obs) >= MIN_COST_PLANETARY_FORTRESS
    )
def pre_call_down_mule(obs):
    orbitals = [u for u in getattr(obs.observation, 'raw_units', [])
                if u.unit_type == units.Terran.OrbitalCommand
                and u.alliance == features.PlayerRelative.SELF
                and getattr(u, 'energy', 0) >= 50]
    return len(orbitals) > 0
def pre_train_ghost(obs):
    return (
        exists(obs, units.Terran.GhostAcademy)
        and supply_available(obs)
        and minerals(obs) >= MIN_COST_GHOST_MINERALS
        and vespene(obs) >= MIN_COST_GHOST_GAS
    )
def pre_add_factory_reactor(obs):
    factories = [
        u for u in getattr(obs.observation, 'feature_units', [])
        if u.unit_type == units.Terran.Factory 
        and u.alliance == features.PlayerRelative.SELF
        and getattr(u, "add_on_tag", 0) == 0
    ]
    return len(factories) > 0 and minerals(obs) >= MIN_COST_REACTOR and vespene(obs) >= 50
def pre_train_hellbat(obs):
    return (
        exists(obs, units.Terran.Factory)
        and exists(obs, units.Terran.Armory)
        and supply_available(obs)
        and minerals(obs) >= 100
    )
def pre_train_widow_mine(obs):
    return (
        exists(obs, units.Terran.Factory)
        and supply_available(obs)
        and minerals(obs) >= MIN_COST_WIDOW_MINE_MINERALS
        and vespene(obs) >= MIN_COST_WIDOW_MINE_GAS
    )
def pre_transform_hellion_to_hellbat(obs):
    return exists(obs, units.Terran.Hellion) and exists(obs, units.Terran.Armory)
def pre_transform_hellbat_to_hellion(obs):
    return exists(obs, units.Terran.Hellbat)
def pre_research_concussive_shells(obs):
    barracks_techlabs = [u for u in getattr(obs.observation, 'raw_units', [])
                         if u.unit_type == units.Terran.BarracksTechLab
                         and u.alliance == features.PlayerRelative.SELF]
    return (
        len(barracks_techlabs) > 0
        and minerals(obs) >= MIN_COST_CONCUSSIVE_SHELLS_MINERALS
        and vespene(obs) >= MIN_COST_CONCUSSIVE_SHELLS_GAS
    )
def pre_research_siege_tech(obs):
    factory_techlabs = [u for u in getattr(obs.observation, 'raw_units', [])
                        if u.unit_type == units.Terran.FactoryTechLab
                        and u.alliance == features.PlayerRelative.SELF]
    return (
        len(factory_techlabs) > 0
        and minerals(obs) >= MIN_COST_SIEGE_TECH_MINERALS
        and vespene(obs) >= MIN_COST_SIEGE_TECH_GAS
    )
def pre_research_infantry_weapons_1(obs):
    return (
        exists(obs, units.Terran.EngineeringBay)
        and minerals(obs) >= MIN_COST_INFANTRY_WEAPONS_MINERALS
        and vespene(obs) >= MIN_COST_INFANTRY_WEAPONS_GAS
    )
def pre_research_infantry_armor_1(obs):
    return (
        exists(obs, units.Terran.EngineeringBay)
        and minerals(obs) >= MIN_COST_INFANTRY_ARMOR_MINERALS
        and vespene(obs) >= MIN_COST_INFANTRY_ARMOR_GAS
    )
def pre_scanner_sweep(obs):
    orbitals = [u for u in getattr(obs.observation, 'raw_units', [])
                if u.unit_type == units.Terran.OrbitalCommand
                and u.alliance == features.PlayerRelative.SELF
                and getattr(u, 'energy', 0) >= 50]
    return len(orbitals) > 0
def pre_siege_tank_siege_mode(obs):
    return exists(obs, units.Terran.SiegeTank)
def pre_siege_tank_unsiege(obs):
    return exists(obs, units.Terran.SiegeTankSieged)
def pre_viking_assault_mode(obs):
    return exists(obs, units.Terran.VikingFighter)
def pre_viking_fighter_mode(obs):
    return exists(obs, units.Terran.VikingAssault)
def pre_burrow_widow_mine(obs):
    return exists(obs, units.Terran.WidowMine)
def pre_unburrow_widow_mine(obs):
    return exists(obs, units.Terran.WidowMineBurrowed)
def pre_build_armory(obs):
    return (
        exists(obs, units.Terran.SCV)
        and exists(obs, units.Terran.Factory)
        and minerals(obs) >= MIN_COST_ARMORY_MINERALS
        and vespene(obs) >= MIN_COST_ARMORY_GAS
        and count_units(obs, units.Terran.Armory) < 1  # MAX 1
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
        and count_units(obs, units.Terran.FusionCore) < 1  # MAX 1
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

TERRAN_STRUCTURES = {
    units.Terran.CommandCenter, units.Terran.OrbitalCommand, units.Terran.PlanetaryFortress,
    units.Terran.Barracks, units.Terran.Factory, units.Terran.Starport,
    units.Terran.SupplyDepot, units.Terran.SupplyDepotLowered,
    units.Terran.Refinery, units.Terran.EngineeringBay, units.Terran.Armory,
    units.Terran.Bunker, units.Terran.MissileTurret, units.Terran.SensorTower,
    units.Terran.GhostAcademy, units.Terran.FusionCore,
    units.Terran.BarracksTechLab, units.Terran.BarracksReactor,
    units.Terran.FactoryTechLab, units.Terran.FactoryReactor,
    units.Terran.StarportTechLab, units.Terran.StarportReactor,
    units.Terran.BarracksFlying, units.Terran.FactoryFlying, units.Terran.StarportFlying,
    units.Terran.CommandCenterFlying,
}

def get_army_supply(obs):
    """Get total supply of combat units"""
    UNIT_SUPPLY = {
        units.Terran.Marine: 1,
        units.Terran.Marauder: 2,
        units.Terran.Reaper: 1,
        units.Terran.Ghost: 2,
        units.Terran.Hellion: 2,
        units.Terran.Hellbat: 2,
        units.Terran.SiegeTank: 3,
        units.Terran.SiegeTankSieged: 3,
        units.Terran.Cyclone: 3,
        units.Terran.Thor: 6,
        units.Terran.WidowMine: 2,
        units.Terran.WidowMineBurrowed: 2,
        units.Terran.VikingFighter: 2,
        units.Terran.VikingAssault: 2,
        units.Terran.Medivac: 2,
        units.Terran.Liberator: 3,
        units.Terran.Raven: 2,
        units.Terran.Banshee: 3,
        units.Terran.Battlecruiser: 6,
    }
    total = 0
    for u in getattr(obs.observation, "raw_units", []):
        if u.alliance == features.PlayerRelative.SELF and u.unit_type in UNIT_SUPPLY:
            total += UNIT_SUPPLY[u.unit_type]
    return total

def pre_has_army(obs):
    """Check if we have any combat units"""
    army_units = [
        u for u in getattr(obs.observation, "raw_units", [])
        if u.alliance == features.PlayerRelative.SELF
        and u.unit_type not in (units.Terran.SCV, units.Terran.MULE)
        and u.unit_type not in TERRAN_STRUCTURES
    ]
    return len(army_units) > 0

def pre_attack_aggressor(obs):
    return get_army_supply(obs) >= 6  # ~6 marines or equivalent

def pre_attack_siege(obs):
    return get_army_supply(obs) >= 12  # Need decent force for siege

def pre_attack_support(obs):
    return get_army_supply(obs) >= 8

def pre_attack_harass(obs):
    return get_army_supply(obs) >= 4  # Small harass ok with 4 supply

def pre_retreat_to_base(obs):
    return pre_has_army(obs)  # Can always retreat

def pre_final_push(obs):
    return get_army_supply(obs) >= 20  # Need real army for final push

def pre_defend(obs):
    return pre_has_army(obs)

def pre_retreat(obs):
    return pre_has_army(obs)

def pre_scout_with_scv(obs):
    return exists(obs, units.Terran.SCV)
def pre_salvage_bunker(obs):
    return exists(obs, units.Terran.Bunker)
def pre_research_barracks_techlab(obs):
    barracks_techlabs = [u for u in getattr(obs.observation, 'raw_units', [])
                         if u.unit_type == units.Terran.BarracksTechLab
                         and u.alliance == features.PlayerRelative.SELF]
    return len(barracks_techlabs) > 0 and minerals(obs) >= 50 and vespene(obs) >= 50

def pre_research_factory_techlab(obs):
    factory_techlabs = [u for u in getattr(obs.observation, 'raw_units', [])
                        if u.unit_type == units.Terran.FactoryTechLab
                        and u.alliance == features.PlayerRelative.SELF]
    return len(factory_techlabs) > 0 and minerals(obs) >= 50 and vespene(obs) >= 50

def pre_research_starport_techlab(obs):
    starport_techlabs = [u for u in getattr(obs.observation, 'raw_units', [])
                         if u.unit_type == units.Terran.StarportTechLab
                         and u.alliance == features.PlayerRelative.SELF]
    return len(starport_techlabs) > 0 and minerals(obs) >= 50 and vespene(obs) >= 50

def pre_research_engineering_bay(obs):
    return (
        exists(obs, units.Terran.EngineeringBay)
        and minerals(obs) >= 100
        and vespene(obs) >= 100
    )

def pre_research_armory(obs):
    return (
        exists(obs, units.Terran.Armory)
        and minerals(obs) >= 100
        and vespene(obs) >= 100
    )

def pre_research_ghost_academy(obs):
    return (
        exists(obs, units.Terran.GhostAcademy)
        and minerals(obs) >= 100
        and vespene(obs) >= 100
    )


ACTION_REQUIREMENTS = {
    "do_nothing": pre_do_nothing,
    "send_idle_workers_to_mine": pre_send_idle_workers_to_mine,
    "send_idle_workers_to_gas": pre_send_idle_workers_to_gas,
    "train_worker": pre_train_worker,
    "build_supply_depot": pre_build_supply_depot,
    "build_command_center": pre_build_command_center,
    "build_barracks": pre_build_barracks,
    "build_factory": pre_build_factory,
    "build_starport": pre_build_starport,
    "build_refinery": pre_build_refinery,
    "build_engineering_bay": pre_build_engineering_bay,
    "build_armory": pre_build_armory,
    "build_fusion_core": pre_build_fusion_core,
    "build_ghost_academy": pre_build_ghost_academy,
    "build_bunker": pre_build_bunker,
    "build_missile_turret": pre_build_missile_turret,
    "build_sensor_tower": pre_build_sensor_tower,
    "add_barracks_techlab": pre_add_barracks_techlab,
    "add_barracks_reactor": pre_add_barracks_reactor,
    "add_factory_techlab": pre_add_factory_techlab,
    "add_factory_reactor": pre_add_factory_reactor,
    "add_starport_techlab": pre_add_starport_techlab,
    "add_starport_reactor": pre_add_starport_reactor,
    "build_techlab": pre_build_techlab,
    "build_reactor": pre_build_reactor,
    "upgrade_orbital_command": pre_upgrade_orbital_command,
    "upgrade_planetary_fortress": pre_upgrade_planetary_fortress,
    "lower_depot": pre_lower_depot,
    "train_marine": pre_train_marine,
    "train_marauder": pre_train_marauder,
    "train_reaper": pre_train_reaper,
    "train_ghost": pre_train_ghost,
    "train_hellion": pre_train_hellion,
    "train_hellbat": pre_train_hellbat,
    "train_siege_tank": pre_train_siege_tank,
    "train_thor": pre_train_thor,
    "train_cyclone": pre_train_cyclone,
    "train_widow_mine": pre_train_widow_mine,
    "train_viking": pre_train_viking,
    "train_medivac": pre_train_medivac,
    "train_liberator": pre_train_liberator,
    "train_raven": pre_train_raven,
    "train_banshee": pre_train_banshee,
    "train_battlecruiser": pre_train_battlecruiser,
    "transform_hellion_to_hellbat": pre_transform_hellion_to_hellbat,
    "transform_hellbat_to_hellion": pre_transform_hellbat_to_hellion,
    "siege_tank_siege_mode": pre_siege_tank_siege_mode,
    "siege_tank_unsiege": pre_siege_tank_unsiege,
    "viking_assault_mode": pre_viking_assault_mode,
    "viking_fighter_mode": pre_viking_fighter_mode,
    "burrow_widow_mine": pre_burrow_widow_mine,
    "unburrow_widow_mine": pre_unburrow_widow_mine,
    "call_down_mule": pre_call_down_mule,
    "scanner_sweep": pre_scanner_sweep,
    "research_barracks_techlab": pre_research_barracks_techlab,
    "research_factory_techlab": pre_research_factory_techlab,
    "research_starport_techlab": pre_research_starport_techlab,
    "research_engineering_bay": pre_research_engineering_bay,
    "research_armory": pre_research_armory,
    "research_ghost_academy": pre_research_ghost_academy,
    "attack_aggressor": pre_attack_aggressor,
    "attack_siege": pre_attack_siege,
    "attack_support": pre_attack_support,
    "attack_harass": pre_attack_harass,
    "retreat_to_base": pre_retreat_to_base,
    "final_push": pre_final_push,
}

def is_action_allowed(action_name, obs):
    func = ACTION_REQUIREMENTS.get(action_name)
    return func(obs) if func else False

def get_legal_actions(obs):
    return [name for name, rule in ACTION_REQUIREMENTS.items() if rule(obs)]