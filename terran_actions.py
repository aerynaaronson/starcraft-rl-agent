"""
Terran action registry and attack logic.
Centralized action management and combat control.

UPDATED: Split attack_move into 6 strategic combat macros
"""

from pysc2.lib import actions, features, units
from terran_helper import is_air_unit, is_ground_unit
from terran_buildings import *
from terran_units import *


# ============================================================================
# BASIC ACTIONS
# ============================================================================

def do_nothing(obs, _x=None, _y=None):
    """Do nothing action"""
    return actions.RAW_FUNCTIONS.no_op()


# ============================================================================
# STRATEGIC COMBAT MACROS
# ============================================================================

def distance(u1, u2):
    """Calculate distance between two units/positions"""
    if isinstance(u1, tuple):
        x1, y1 = u1
    else:
        x1, y1 = u1.x, u1.y
    
    if isinstance(u2, tuple):
        x2, y2 = u2
    else:
        x2, y2 = u2.x, u2.y
    
    return ((x1 - x2)**2 + (y1 - y2)**2)**0.5


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
}

def get_army_units(obs):
    """Get all combat units (exclude workers/structures)"""
    return [
        u for u in obs.observation.raw_units
        if u.alliance == features.PlayerRelative.SELF
        and u.unit_type not in (units.Terran.SCV, units.Terran.MULE)
        and u.unit_type not in TERRAN_STRUCTURES
    ]


def get_enemy_units(obs):
    """Get all enemy units"""
    return [u for u in obs.observation.raw_units if u.alliance == features.PlayerRelative.ENEMY]


def get_closest_command_center(obs, position):
    """Find closest CC to a position"""
    ccs = [
        u for u in obs.observation.raw_units
        if u.alliance == features.PlayerRelative.SELF
        and u.unit_type in [units.Terran.CommandCenter, units.Terran.OrbitalCommand, units.Terran.PlanetaryFortress]
    ]
    if not ccs:
        return None
    return min(ccs, key=lambda cc: distance(position, (cc.x, cc.y)))


def get_enemy_base_location(obs):
    """Estimate enemy base location from known buildings/units"""
    enemies = get_enemy_units(obs)
    if not enemies:
        all_units = obs.observation.raw_units
        if len(all_units) > 0:
            max_x = max(u.x for u in all_units)
            max_y = max(u.y for u in all_units)
            return (int(max_x * 0.75), int(max_y * 0.75))
        return (100, 100)
    
    avg_x = sum(e.x for e in enemies) / len(enemies)
    avg_y = sum(e.y for e in enemies) / len(enemies)
    return (int(avg_x), int(avg_y))


def attack_aggressor(obs, _x=None, _y=None):
    """
    AGGRESSOR: Frontline push with bio/mech assault units
    Uses: Marines, Marauders, Hellbats, Thors, Battlecruisers
    """
    army = get_army_units(obs)
    if not army:
        return actions.RAW_FUNCTIONS.no_op()
    
    enemies = get_enemy_units(obs)
    
    bio_aggro = [u for u in army if u.unit_type in [units.Terran.Marine, units.Terran.Marauder]]
    mech_aggro = [u for u in army if u.unit_type in [units.Terran.Hellbat, units.Terran.Thor]]
    air_aggro = [u for u in army if u.unit_type == units.Terran.Battlecruiser]
    
    action_list = []
    
    # BIO: Stim and charge
    if bio_aggro:
        bio_tags = [u.tag for u in bio_aggro]
        stim_tags = [u.tag for u in bio_aggro if u.energy >= 10]
        
        if stim_tags and enemies:
            closest_dist = min(distance(bio_aggro[0], e) for e in enemies)
            if closest_dist < 20:
                action_list.append(actions.RAW_FUNCTIONS.Effect_Stim_quick("now", stim_tags))
        
        if enemies:
            target = min(enemies, key=lambda e: distance(bio_aggro[0], e))
            action_list.append(actions.RAW_FUNCTIONS.Attack_pt("now", bio_tags, (int(target.x), int(target.y))))
        else:
            enemy_base = get_enemy_base_location(obs)
            action_list.append(actions.RAW_FUNCTIONS.Attack_pt("now", bio_tags, enemy_base))
    
    # MECH: Push forward
    if mech_aggro:
        mech_tags = [u.tag for u in mech_aggro]
        if enemies:
            target = min(enemies, key=lambda e: distance(mech_aggro[0], e))
            action_list.append(actions.RAW_FUNCTIONS.Attack_pt("now", mech_tags, (int(target.x), int(target.y))))
        else:
            enemy_base = get_enemy_base_location(obs)
            action_list.append(actions.RAW_FUNCTIONS.Attack_pt("now", mech_tags, enemy_base))
    
    # AIR: Battlecruisers
    if air_aggro:
        air_tags = [u.tag for u in air_aggro]
        for bc in air_aggro:
            if bc.energy >= 125 and enemies:
                target = min(enemies, key=lambda e: distance(bc, e))
                if distance(bc, target) <= 10:
                    action_list.append(actions.RAW_FUNCTIONS.Effect_YamatoGun_screen("now", [bc.tag], (int(target.x), int(target.y))))
                    break
        
        if enemies:
            target = min(enemies, key=lambda e: distance(air_aggro[0], e))
            action_list.append(actions.RAW_FUNCTIONS.Attack_pt("now", air_tags, (int(target.x), int(target.y))))
        else:
            enemy_base = get_enemy_base_location(obs)
            action_list.append(actions.RAW_FUNCTIONS.Attack_pt("now", air_tags, enemy_base))
    
    return action_list[0] if action_list else actions.RAW_FUNCTIONS.no_op()


def attack_siege(obs, _x=None, _y=None):
    """
    SIEGE: Position siege units for area control
    Uses: Siege Tanks, Liberators, Widow Mines
    """
    army = get_army_units(obs)
    if not army:
        return actions.RAW_FUNCTIONS.no_op()
    
    enemies = get_enemy_units(obs)
    
    tanks = [u for u in army if u.unit_type in [units.Terran.SiegeTank, units.Terran.SiegeTankSieged]]
    liberators = [u for u in army if u.unit_type in [units.Terran.Liberator, units.Terran.LiberatorAG]]
    mines = [u for u in army if u.unit_type in [units.Terran.WidowMine, units.Terran.WidowMineBurrowed]]
    
    action_list = []
    
    # TANKS: Siege management
    for tank in tanks:
        if tank.unit_type == units.Terran.SiegeTankSieged:
            if enemies:
                closest = min(enemies, key=lambda e: distance(tank, e))
                dist = distance(tank, closest)
                if dist < 5 or dist > 13:
                    action_list.append(actions.RAW_FUNCTIONS.Morph_Unsiege_quick("now", [tank.tag]))
                else:
                    action_list.append(actions.RAW_FUNCTIONS.Attack_pt("now", [tank.tag], (int(closest.x), int(closest.y))))
            else:
                action_list.append(actions.RAW_FUNCTIONS.Morph_Unsiege_quick("now", [tank.tag]))
        else:
            if enemies:
                closest = min(enemies, key=lambda e: distance(tank, e))
                dist = distance(tank, closest)
                if 7 <= dist <= 13:
                    action_list.append(actions.RAW_FUNCTIONS.Morph_SiegeMode_quick("now", [tank.tag]))
                elif dist > 13:
                    action_list.append(actions.RAW_FUNCTIONS.Attack_pt("now", [tank.tag], (int(closest.x), int(closest.y))))
                else:
                    retreat_x = tank.x + (tank.x - closest.x) * 0.3
                    retreat_y = tank.y + (tank.y - closest.y) * 0.3
                    action_list.append(actions.RAW_FUNCTIONS.Move_pt("now", [tank.tag], (int(retreat_x), int(retreat_y))))
            else:
                enemy_base = get_enemy_base_location(obs)
                action_list.append(actions.RAW_FUNCTIONS.Attack_pt("now", [tank.tag], enemy_base))
    
    # LIBERATORS: Siege mode switching
    for lib in liberators:
        ground_enemies = [e for e in enemies if not is_air_unit(e)]
        air_enemies = [e for e in enemies if is_air_unit(e)]
        
        if lib.unit_type == units.Terran.LiberatorAG:
            if air_enemies or not ground_enemies:
                action_list.append(actions.RAW_FUNCTIONS.Morph_LiberatorAAMode_quick("now", [lib.tag]))
            elif ground_enemies:
                target = min(ground_enemies, key=lambda e: distance(lib, e))
                action_list.append(actions.RAW_FUNCTIONS.Attack_pt("now", [lib.tag], (int(target.x), int(target.y))))
        else:
            if ground_enemies and not air_enemies:
                target = min(ground_enemies, key=lambda e: distance(lib, e))
                if distance(lib, target) <= 12:
                    action_list.append(actions.RAW_FUNCTIONS.Morph_LiberatorAGMode_pt("now", [lib.tag], (int(target.x), int(target.y))))
                else:
                    action_list.append(actions.RAW_FUNCTIONS.Move_pt("now", [lib.tag], (int(target.x), int(target.y))))
            elif enemies:
                target = min(enemies, key=lambda e: distance(lib, e))
                action_list.append(actions.RAW_FUNCTIONS.Attack_pt("now", [lib.tag], (int(target.x), int(target.y))))
            else:
                enemy_base = get_enemy_base_location(obs)
                action_list.append(actions.RAW_FUNCTIONS.Move_pt("now", [lib.tag], enemy_base))
    
    # WIDOW MINES: Burrow
    for mine in mines:
        if mine.unit_type == units.Terran.WidowMine:
            if enemies:
                closest = min(enemies, key=lambda e: distance(mine, e))
                if distance(mine, closest) > 6:
                    action_list.append(actions.RAW_FUNCTIONS.BurrowDown_WidowMine_quick("now", [mine.tag]))
                else:
                    retreat_x = mine.x + (mine.x - closest.x)
                    retreat_y = mine.y + (mine.y - closest.y)
                    action_list.append(actions.RAW_FUNCTIONS.Move_pt("now", [mine.tag], (int(retreat_x), int(retreat_y))))
            else:
                action_list.append(actions.RAW_FUNCTIONS.BurrowDown_WidowMine_quick("now", [mine.tag]))
    
    return action_list[0] if action_list else actions.RAW_FUNCTIONS.no_op()


def attack_support(obs, _x=None, _y=None):
    """
    SUPPORT: Utility units with abilities
    Uses: Medivacs, Ravens, Vikings, Ghosts
    """
    army = get_army_units(obs)
    if not army:
        return actions.RAW_FUNCTIONS.no_op()
    
    enemies = get_enemy_units(obs)
    
    medivacs = [u for u in army if u.unit_type == units.Terran.Medivac]
    ravens = [u for u in army if u.unit_type == units.Terran.Raven]
    vikings = [u for u in army if u.unit_type in [units.Terran.VikingFighter, units.Terran.VikingAssault]]
    ghosts = [u for u in army if u.unit_type == units.Terran.Ghost]
    
    action_list = []
    
    if army:
        army_x = sum(u.x for u in army) / len(army)
        army_y = sum(u.y for u in army) / len(army)
        army_center = (army_x, army_y)
    else:
        army_center = (50, 50)
    
    # MEDIVACS: Follow army
    if medivacs:
        medivac_tags = [m.tag for m in medivacs]
        action_list.append(actions.RAW_FUNCTIONS.Move_pt("now", medivac_tags, (int(army_center[0]), int(army_center[1]))))
    
    # RAVENS: Stay back
    if ravens:
        for raven in ravens:
            if enemies:
                closest = min(enemies, key=lambda e: distance(raven, e))
                retreat_x = army_center[0] + (army_center[0] - closest.x) * 0.2
                retreat_y = army_center[1] + (army_center[1] - closest.y) * 0.2
                action_list.append(actions.RAW_FUNCTIONS.Move_pt("now", [raven.tag], (int(retreat_x), int(retreat_y))))
            else:
                action_list.append(actions.RAW_FUNCTIONS.Move_pt("now", [raven.tag], (int(army_center[0]), int(army_center[1]))))
    
    # VIKINGS: Transform based on threats
    for viking in vikings:
        air_enemies = [e for e in enemies if is_air_unit(e)]
        ground_enemies = [e for e in enemies if not is_air_unit(e)]
        
        if viking.unit_type == units.Terran.VikingFighter:
            if air_enemies:
                target = min(air_enemies, key=lambda e: distance(viking, e))
                action_list.append(actions.RAW_FUNCTIONS.Attack_pt("now", [viking.tag], (int(target.x), int(target.y))))
            elif ground_enemies:
                action_list.append(actions.RAW_FUNCTIONS.Morph_VikingAssaultMode_quick("now", [viking.tag]))
            else:
                action_list.append(actions.RAW_FUNCTIONS.Move_pt("now", [viking.tag], (int(army_center[0]), int(army_center[1]))))
        else:
            if air_enemies:
                action_list.append(actions.RAW_FUNCTIONS.Morph_VikingFighterMode_quick("now", [viking.tag]))
            elif ground_enemies:
                target = min(ground_enemies, key=lambda e: distance(viking, e))
                action_list.append(actions.RAW_FUNCTIONS.Attack_pt("now", [viking.tag], (int(target.x), int(target.y))))
            else:
                action_list.append(actions.RAW_FUNCTIONS.Morph_VikingFighterMode_quick("now", [viking.tag]))
    
    # GHOSTS: Follow army
    if ghosts:
        for ghost in ghosts:
            if enemies:
                target = min(enemies, key=lambda e: distance(ghost, e))
                action_list.append(actions.RAW_FUNCTIONS.Attack_pt("now", [ghost.tag], (int(target.x), int(target.y))))
            else:
                action_list.append(actions.RAW_FUNCTIONS.Move_pt("now", [ghost.tag], (int(army_center[0]), int(army_center[1]))))
    
    return action_list[0] if action_list else actions.RAW_FUNCTIONS.no_op()


def attack_harass(obs, _x=None, _y=None):
    """
    HARASS: Fast raiders for map control
    Uses: Reapers, Hellions, Banshees, Cyclones
    """
    army = get_army_units(obs)
    if not army:
        return actions.RAW_FUNCTIONS.no_op()
    
    enemies = get_enemy_units(obs)
    
    reapers = [u for u in army if u.unit_type == units.Terran.Reaper]
    hellions = [u for u in army if u.unit_type == units.Terran.Hellion]
    banshees = [u for u in army if u.unit_type == units.Terran.Banshee]
    cyclones = [u for u in army if u.unit_type == units.Terran.Cyclone]
    
    action_list = []
    enemy_base = get_enemy_base_location(obs)
    
    # REAPERS: Worker harass
    if reapers:
        for reaper in reapers:
            if enemies:
                target = min(enemies, key=lambda e: distance(reaper, e))
                if reaper.energy >= 50:
                    nearby = [e for e in enemies if distance(reaper, e) < 8]
                    if len(nearby) >= 3:
                        action_list.append(actions.RAW_FUNCTIONS.Effect_KD8Charge_screen("now", [reaper.tag], (int(target.x), int(target.y))))
                        continue
                action_list.append(actions.RAW_FUNCTIONS.Attack_pt("now", [reaper.tag], (int(target.x), int(target.y))))
            else:
                action_list.append(actions.RAW_FUNCTIONS.Attack_pt("now", [reaper.tag], enemy_base))
    
    # HELLIONS: Runbys
    if hellions:
        hellion_tags = [h.tag for h in hellions]
        if enemies:
            target = min(enemies, key=lambda e: distance(hellions[0], e))
            action_list.append(actions.RAW_FUNCTIONS.Attack_pt("now", hellion_tags, (int(target.x), int(target.y))))
        else:
            action_list.append(actions.RAW_FUNCTIONS.Attack_pt("now", hellion_tags, enemy_base))
    
    # BANSHEES: Cloak and attack
    if banshees:
        for banshee in banshees:
            if banshee.energy >= 25 and enemies:
                closest = min(enemies, key=lambda e: distance(banshee, e))
                if distance(banshee, closest) < 15:
                    action_list.append(actions.RAW_FUNCTIONS.Behavior_CloakOn_Banshee_quick("now", [banshee.tag]))
            
            if enemies:
                target = min(enemies, key=lambda e: distance(banshee, e))
                action_list.append(actions.RAW_FUNCTIONS.Attack_pt("now", [banshee.tag], (int(target.x), int(target.y))))
            else:
                action_list.append(actions.RAW_FUNCTIONS.Attack_pt("now", [banshee.tag], enemy_base))
    
    # CYCLONES: Lock-on
    if cyclones:
        for cyclone in cyclones:
            if enemies:
                target = min(enemies, key=lambda e: distance(cyclone, e))
                if cyclone.energy >= 50 and distance(cyclone, target) <= 15:
                    action_list.append(actions.RAW_FUNCTIONS.Effect_LockOn_screen("now", [cyclone.tag], (int(target.x), int(target.y))))
                else:
                    action_list.append(actions.RAW_FUNCTIONS.Attack_pt("now", [cyclone.tag], (int(target.x), int(target.y))))
            else:
                action_list.append(actions.RAW_FUNCTIONS.Attack_pt("now", [cyclone.tag], enemy_base))
    
    return action_list[0] if action_list else actions.RAW_FUNCTIONS.no_op()


def retreat_to_base(obs, _x=None, _y=None):
    """
    RETREAT: Fall back all units to nearest CC
    """
    army = get_army_units(obs)
    if not army:
        return actions.RAW_FUNCTIONS.no_op()
    
    army_x = sum(u.x for u in army) / len(army)
    army_y = sum(u.y for u in army) / len(army)
    army_center = (army_x, army_y)
    
    closest_cc = get_closest_command_center(obs, army_center)
    if closest_cc is None:
        return actions.RAW_FUNCTIONS.no_op()
    
    rally_point = (int(closest_cc.x), int(closest_cc.y))
    army_tags = [u.tag for u in army]
    return actions.RAW_FUNCTIONS.Move_pt("now", army_tags, rally_point)


def final_push(obs, _x=None, _y=None):
    """
    FINAL PUSH: All-in attack on enemy base
    """
    army = get_army_units(obs)
    if not army:
        return actions.RAW_FUNCTIONS.no_op()
    
    enemies = get_enemy_units(obs)
    
    if enemies:
        target_x = sum(e.x for e in enemies) / len(enemies)
        target_y = sum(e.y for e in enemies) / len(enemies)
        target = (int(target_x), int(target_y))
    else:
        target = get_enemy_base_location(obs)
    
    army_tags = [u.tag for u in army]
    return actions.RAW_FUNCTIONS.Attack_pt("now", army_tags, target)


# ============================================================================
# ACTION REGISTRY - Maps action names to functions
# ============================================================================

ACTION_REGISTRY = {
    # Basic actions
    "do_nothing": do_nothing,
    
    # Worker management
    "send_idle_workers_to_mine": send_idle_workers_to_mine,
    "send_idle_workers_to_gas": send_idle_workers_to_gas,
    "train_worker": train_worker,
    
    # Buildings - Core
    "build_supply_depot": build_supply_depot,
    "build_command_center": build_command_center,
    "build_barracks": build_barracks,
    "build_factory": build_factory,
    "build_starport": build_starport,
    "build_refinery": build_refinery,
    
    # Buildings - Tech
    "build_engineering_bay": build_engineering_bay,
    "build_armory": build_armory,
    "build_fusion_core": build_fusion_core,
    "build_ghost_academy": build_ghost_academy,
    
    # Buildings - Defensive
    "build_bunker": build_bunker,
    "build_missile_turret": build_missile_turret,
    "build_sensor_tower": build_sensor_tower,
    
    # Addons
    "add_barracks_techlab": add_barracks_techlab,
    "add_barracks_reactor": add_barracks_reactor,
    "add_factory_techlab": add_factory_techlab,
    "add_factory_reactor": add_factory_reactor,
    "add_starport_techlab": add_starport_techlab,
    "add_starport_reactor": add_starport_reactor,
    "add_barracks_addon": add_barracks_addon,
    "add_factory_addon": add_factory_addon,
    "add_starport_addon": add_starport_addon,
    "build_techlab": build_techlab,
    "build_reactor": build_reactor,
    
    # Building upgrades
    "upgrade_orbital_command": upgrade_orbital_command,
    "upgrade_planetary_fortress": upgrade_planetary_fortress,
    "lower_depot": lower_depot,
    
    # Barracks units
    "train_marine": train_marine,
    "train_marauder": train_marauder,
    "train_reaper": train_reaper,
    "train_ghost": train_ghost,
    
    # Factory units
    "train_hellion": train_hellion,
    "train_hellbat": train_hellbat,
    "train_siege_tank": train_siege_tank,
    "train_thor": train_thor,
    "train_cyclone": train_cyclone,
    "train_widow_mine": train_widow_mine,
    
    # Starport units
    "train_viking": train_viking,
    "train_medivac": train_medivac,
    "train_liberator": train_liberator,
    "train_raven": train_raven,
    "train_banshee": train_banshee,
    "train_battlecruiser": train_battlecruiser,
    
    # Abilities
    "call_down_mule": call_down_mule,
    "scanner_sweep": scanner_sweep,
    
    # Research
    "research_barracks_techlab": research_barracks_techlab,
    "research_factory_techlab": research_factory_techlab,
    "research_starport_techlab": research_starport_techlab,
    "research_engineering_bay": research_engineering_bay,
    "research_armory": research_armory,
    "research_ghost_academy": research_ghost_academy,
    
    # Combat - NEW STRATEGIC MACROS
    "attack_aggressor": attack_aggressor,
    "attack_siege": attack_siege,
    "attack_support": attack_support,
    "attack_harass": attack_harass,
    "retreat_to_base": retreat_to_base,
    "final_push": final_push,
}


# ============================================================================
# ACTION NEEDS COORDS - Which actions require coordinate parameters
# ============================================================================

ACTION_NEEDS_COORDS = {
    # Basic actions
    "do_nothing": False,
    
    # Worker management
    "send_idle_workers_to_mine": False,
    "send_idle_workers_to_gas": False,
    "train_worker": False,
    
    # Buildings - Core
    "build_supply_depot": True,
    "build_command_center": False,
    "build_barracks": True,
    "build_factory": True,
    "build_starport": True,
    "build_refinery": False,
    
    # Buildings - Tech
    "build_engineering_bay": True,
    "build_armory": True,
    "build_fusion_core": True,
    "build_ghost_academy": True,
    
    # Buildings - Defensive
    "build_bunker": True,
    "build_missile_turret": True,
    "build_sensor_tower": True,
    
    # Addons
    "add_barracks_addon": False,
    "add_factory_addon": False,
    "add_starport_addon": False,
    
    # Building upgrades
    "upgrade_orbital_command": False,
    "upgrade_planetary_fortress": False,
    "lower_depot": False,
    
    # Barracks units
    "train_marine": False,
    "train_marauder": False,
    "train_reaper": False,
    "train_ghost": False,
    
    # Factory units
    "train_hellion": False,
    "train_hellbat": False,
    "train_siege_tank": False,
    "train_thor": False,
    "train_cyclone": False,
    "train_widow_mine": False,
    
    # Starport units
    "train_viking": False,
    "train_medivac": False,
    "train_liberator": False,
    "train_raven": False,
    "train_banshee": False,
    "train_battlecruiser": False,
    
    # Abilities
    "call_down_mule": True,
    "scanner_sweep": True,
    
    # Research
    "research_barracks_techlab": False,
    "research_factory_techlab": False,
    "research_starport_techlab": False,
    "research_engineering_bay": False,
    "research_armory": False,
    "research_ghost_academy": False,
    
    # Combat - NEW (all False, they figure out targets themselves)
    "attack_aggressor": False,
    "attack_siege": False,
    "attack_support": False,
    "attack_harass": False,
    "retreat_to_base": False,
    "final_push": False,
}