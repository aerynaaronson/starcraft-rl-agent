"""
Terran action registry and attack logic.
Centralized action management and combat control.

UPDATED: Combat macros use network coordinates for independent army control
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


def attack_aggressor(obs, target_x=None, target_y=None):
    """
    AGGRESSOR: Frontline push with bio/mech assault units
    Uses: Marines, Marauders, Hellbats, Thors, Battlecruisers
    
    Target coordinates from network determine WHERE to push.
    """
    army = get_army_units(obs)
    if not army:
        return actions.RAW_FUNCTIONS.no_op()
    
    # Get aggressor units
    bio_aggro = [u for u in army if u.unit_type in [units.Terran.Marine, units.Terran.Marauder]]
    mech_aggro = [u for u in army if u.unit_type in [units.Terran.Hellbat, units.Terran.Thor]]
    air_aggro = [u for u in army if u.unit_type == units.Terran.Battlecruiser]
    
    aggro_units = bio_aggro + mech_aggro + air_aggro
    if not aggro_units:
        return actions.RAW_FUNCTIONS.no_op()
    
    # Determine target location
    if target_x is not None and target_y is not None:
        target = (int(target_x), int(target_y))
    else:
        # Fallback: attack closest enemy or enemy base
        enemies = get_enemy_units(obs)
        if enemies:
            target = (int(enemies[0].x), int(enemies[0].y))
        else:
            target = get_enemy_base_location(obs)
    
    enemies = get_enemy_units(obs)
    action_list = []
    
    # BIO: Stim if enemies nearby target
    if bio_aggro:
        bio_tags = [u.tag for u in bio_aggro]
        stim_candidates = [u for u in bio_aggro if u.energy >= 10]
        
        if stim_candidates and enemies:
            # Check if enemies near our target
            enemies_near_target = [e for e in enemies if distance(target, (e.x, e.y)) < 15]
            if enemies_near_target:
                closest_to_bio = min(enemies, key=lambda e: distance(bio_aggro[0], e))
                if distance(bio_aggro[0], closest_to_bio) < 20:
                    action_list.append(actions.RAW_FUNCTIONS.Effect_Stim_quick("now", [u.tag for u in stim_candidates]))
        
        action_list.append(actions.RAW_FUNCTIONS.Attack_pt("now", bio_tags, target))
    
    # MECH: Push to target
    if mech_aggro:
        mech_tags = [u.tag for u in mech_aggro]
        action_list.append(actions.RAW_FUNCTIONS.Attack_pt("now", mech_tags, target))
    
    # AIR: Battlecruisers to target, Yamato high-value if in range
    if air_aggro:
        air_tags = [u.tag for u in air_aggro]
        for bc in air_aggro:
            if bc.energy >= 125 and enemies:
                # Yamato closest enemy to our target
                enemies_near = [e for e in enemies if distance(target, (e.x, e.y)) < 20]
                if enemies_near:
                    yamato_target = min(enemies_near, key=lambda e: distance(bc, e))
                    if distance(bc, yamato_target) <= 10:
                        action_list.append(actions.RAW_FUNCTIONS.Effect_YamatoGun_unit("now", bc.tag, yamato_target.tag))
                        break
        
        action_list.append(actions.RAW_FUNCTIONS.Attack_pt("now", air_tags, target))
    
    return action_list[0] if action_list else actions.RAW_FUNCTIONS.no_op()


def attack_siege(obs, target_x=None, target_y=None):
    """
    SIEGE: Position siege units for area control
    Uses: Siege Tanks, Liberators, Widow Mines
    
    Target coordinates from network determine WHERE to set up siege.
    """
    army = get_army_units(obs)
    if not army:
        return actions.RAW_FUNCTIONS.no_op()
    
    tanks = [u for u in army if u.unit_type in [units.Terran.SiegeTank, units.Terran.SiegeTankSieged]]
    liberators = [u for u in army if u.unit_type in [units.Terran.Liberator, units.Terran.LiberatorAG]]
    mines = [u for u in army if u.unit_type in [units.Terran.WidowMine, units.Terran.WidowMineBurrowed]]
    
    siege_units = tanks + liberators + mines
    if not siege_units:
        return actions.RAW_FUNCTIONS.no_op()
    
    # Determine target location for siege setup
    if target_x is not None and target_y is not None:
        target = (int(target_x), int(target_y))
    else:
        # Fallback: position toward enemy
        target = get_enemy_base_location(obs)
    
    enemies = get_enemy_units(obs)
    action_list = []
    
    # TANKS: Move to target, siege when in position
    for tank in tanks:
        dist_to_target = distance(tank, target)
        
        if tank.unit_type == units.Terran.SiegeTankSieged:
            # Already sieged - check if we should unsiege to reposition
            if dist_to_target > 15:
                # Too far from target, unsiege to move
                action_list.append(actions.RAW_FUNCTIONS.Morph_Unsiege_quick("now", [tank.tag]))
            elif enemies:
                # Sieged and in position - attack enemies in range
                in_range = [e for e in enemies if distance(tank, e) <= 13]
                if in_range:
                    closest = min(in_range, key=lambda e: distance(tank, e))
                    action_list.append(actions.RAW_FUNCTIONS.Attack_pt("now", [tank.tag], (int(closest.x), int(closest.y))))
                elif distance(tank, target) > 13:
                    # No enemies in range and not at target
                    action_list.append(actions.RAW_FUNCTIONS.Morph_Unsiege_quick("now", [tank.tag]))
        else:
            # Unsieged tank
            if dist_to_target <= 8:
                # At target position, siege up
                action_list.append(actions.RAW_FUNCTIONS.Morph_SiegeMode_quick("now", [tank.tag]))
            else:
                # Move toward target
                action_list.append(actions.RAW_FUNCTIONS.Move_pt("now", [tank.tag], target))
    
    # LIBERATORS: Set up at target
    for lib in liberators:
        dist_to_target = distance(lib, target)
        ground_enemies = [e for e in enemies if not is_air_unit(e)]
        
        if lib.unit_type == units.Terran.LiberatorAG:
            # Already in AG mode
            if dist_to_target > 15:
                # Too far, switch to AA and reposition
                action_list.append(actions.RAW_FUNCTIONS.Morph_LiberatorAAMode_quick("now", [lib.tag]))
        else:
            # AA mode
            if dist_to_target <= 10 and ground_enemies:
                # At target with ground enemies, switch to AG
                action_list.append(actions.RAW_FUNCTIONS.Morph_LiberatorAGMode_pt("now", [lib.tag], target))
            else:
                # Move toward target
                action_list.append(actions.RAW_FUNCTIONS.Move_pt("now", [lib.tag], target))
    
    # WIDOW MINES: Burrow at target
    for mine in mines:
        dist_to_target = distance(mine, target)
        
        if mine.unit_type == units.Terran.WidowMine:
            if dist_to_target <= 5:
                # At target, burrow
                action_list.append(actions.RAW_FUNCTIONS.BurrowDown_WidowMine_quick("now", [mine.tag]))
            else:
                # Move to target
                action_list.append(actions.RAW_FUNCTIONS.Move_pt("now", [mine.tag], target))
        # Burrowed mines stay put
    
    return action_list[0] if action_list else actions.RAW_FUNCTIONS.no_op()


def attack_support(obs, target_x=None, target_y=None):
    """
    SUPPORT: Utility units with abilities
    Uses: Medivacs, Ravens, Vikings, Ghosts
    
    Target coordinates from network determine WHERE to position support.
    """
    army = get_army_units(obs)
    if not army:
        return actions.RAW_FUNCTIONS.no_op()
    
    medivacs = [u for u in army if u.unit_type == units.Terran.Medivac]
    ravens = [u for u in army if u.unit_type == units.Terran.Raven]
    vikings = [u for u in army if u.unit_type in [units.Terran.VikingFighter, units.Terran.VikingAssault]]
    ghosts = [u for u in army if u.unit_type == units.Terran.Ghost]
    
    support_units = medivacs + ravens + vikings + ghosts
    if not support_units:
        return actions.RAW_FUNCTIONS.no_op()
    
    # Determine target location
    if target_x is not None and target_y is not None:
        target = (int(target_x), int(target_y))
    else:
        # Fallback: follow army center
        if army:
            army_x = sum(u.x for u in army) / len(army)
            army_y = sum(u.y for u in army) / len(army)
            target = (int(army_x), int(army_y))
        else:
            target = (50, 50)
    
    enemies = get_enemy_units(obs)
    action_list = []
    
    # MEDIVACS: Move to target (will auto-heal nearby bio)
    if medivacs:
        medivac_tags = [m.tag for m in medivacs]
        action_list.append(actions.RAW_FUNCTIONS.Move_pt("now", medivac_tags, target))
    
    # RAVENS: Position at target, stay back from enemies
    if ravens:
        for raven in ravens:
            # Check for enemies near target
            enemies_near = [e for e in enemies if distance(target, (e.x, e.y)) < 15]
            if enemies_near:
                # Stay slightly back from target
                closest_enemy = min(enemies_near, key=lambda e: distance(target, (e.x, e.y)))
                dx = target[0] - closest_enemy.x
                dy = target[1] - closest_enemy.y
                safe_x = target[0] + dx * 0.3
                safe_y = target[1] + dy * 0.3
                action_list.append(actions.RAW_FUNCTIONS.Move_pt("now", [raven.tag], (int(safe_x), int(safe_y))))
            else:
                action_list.append(actions.RAW_FUNCTIONS.Move_pt("now", [raven.tag], target))
    
    # VIKINGS: Move to target, transform based on threats
    for viking in vikings:
        air_enemies = [e for e in enemies if is_air_unit(e)]
        
        if viking.unit_type == units.Terran.VikingFighter:
            if air_enemies:
                closest_air = min(air_enemies, key=lambda e: distance(viking, e))
                action_list.append(actions.RAW_FUNCTIONS.Attack_pt("now", [viking.tag], (int(closest_air.x), int(closest_air.y))))
            else:
                # No air threats, move to target (or transform if ground enemies)
                ground_near_target = [e for e in enemies if not is_air_unit(e) and distance(target, (e.x, e.y)) < 15]
                if ground_near_target:
                    action_list.append(actions.RAW_FUNCTIONS.Morph_VikingAssaultMode_quick("now", [viking.tag]))
                else:
                    action_list.append(actions.RAW_FUNCTIONS.Move_pt("now", [viking.tag], target))
        else:
            # Assault mode
            if air_enemies:
                action_list.append(actions.RAW_FUNCTIONS.Morph_VikingFighterMode_quick("now", [viking.tag]))
            else:
                action_list.append(actions.RAW_FUNCTIONS.Attack_pt("now", [viking.tag], target))
    
    # GHOSTS: Move to target, use abilities
    if ghosts:
        for ghost in ghosts:
            action_list.append(actions.RAW_FUNCTIONS.Attack_pt("now", [ghost.tag], target))
    
    return action_list[0] if action_list else actions.RAW_FUNCTIONS.no_op()


def attack_harass(obs, target_x=None, target_y=None):
    """
    HARASS: Fast raiders for economic damage
    Uses: Reapers, Hellions, Banshees, Cyclones
    
    Target coordinates from network determine WHERE to raid (e.g. enemy mineral line).
    """
    army = get_army_units(obs)
    if not army:
        return actions.RAW_FUNCTIONS.no_op()
    
    reapers = [u for u in army if u.unit_type == units.Terran.Reaper]
    hellions = [u for u in army if u.unit_type == units.Terran.Hellion]
    banshees = [u for u in army if u.unit_type == units.Terran.Banshee]
    cyclones = [u for u in army if u.unit_type == units.Terran.Cyclone]
    
    harass_units = reapers + hellions + banshees + cyclones
    if not harass_units:
        return actions.RAW_FUNCTIONS.no_op()
    
    # Determine target location for harassment
    if target_x is not None and target_y is not None:
        target = (int(target_x), int(target_y))
    else:
        # Fallback: enemy base
        target = get_enemy_base_location(obs)
    
    enemies = get_enemy_units(obs)
    action_list = []
    
    # REAPERS: Hit and run at target
    if reapers:
        for reaper in reapers:
            # Use KD8 grenade if enemies clustered near target
            if reaper.energy >= 50:
                enemies_at_target = [e for e in enemies if distance(target, (e.x, e.y)) < 8]
                if len(enemies_at_target) >= 3:
                    action_list.append(actions.RAW_FUNCTIONS.Effect_KD8Charge_pt("now", [reaper.tag], target))
                    continue
            
            action_list.append(actions.RAW_FUNCTIONS.Attack_pt("now", [reaper.tag], target))
    
    # HELLIONS: Runby to target
    if hellions:
        hellion_tags = [h.tag for h in hellions]
        action_list.append(actions.RAW_FUNCTIONS.Attack_pt("now", hellion_tags, target))
    
    # BANSHEES: Cloak and hit target
    if banshees:
        for banshee in banshees:
            # Cloak when approaching enemies
            enemies_near_target = [e for e in enemies if distance(target, (e.x, e.y)) < 15]
            if banshee.energy >= 25 and enemies_near_target and distance(banshee, target) < 20:
                action_list.append(actions.RAW_FUNCTIONS.Behavior_CloakOn_Banshee_quick("now", [banshee.tag]))
            
            action_list.append(actions.RAW_FUNCTIONS.Attack_pt("now", [banshee.tag], target))
    
    # CYCLONES: Lock-on priority targets at target location
    if cyclones:
        for cyclone in cyclones:
            enemies_at_target = [e for e in enemies if distance(target, (e.x, e.y)) < 15]
            if cyclone.energy >= 50 and enemies_at_target:
                lock_target = enemies_at_target[0]
                if distance(cyclone, lock_target) <= 15:
                    action_list.append(actions.RAW_FUNCTIONS.Effect_LockOn_unit("now", cyclone.tag, lock_target.tag))
                    continue
            
            action_list.append(actions.RAW_FUNCTIONS.Attack_pt("now", [cyclone.tag], target))
    
    return action_list[0] if action_list else actions.RAW_FUNCTIONS.no_op()


def retreat_to_base(obs, _x=None, _y=None):
    """
    RETREAT: Fall back ALL units to nearest CC.
    Does NOT use coordinates - always goes home.
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
    FINAL PUSH: All-in attack on enemy base.
    Does NOT use coordinates - always goes to enemy.
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
    
    # Combat - COORDINATE-BASED MACROS
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
    "add_barracks_techlab": False,
    "add_barracks_reactor": False,
    "add_factory_techlab": False,
    "add_factory_reactor": False,
    "add_starport_techlab": False,
    "add_starport_reactor": False,
    "build_techlab": False,
    "build_reactor": False,
    
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
    
    # Combat - NOW USE COORDINATES (except retreat/final_push)
    "attack_aggressor": True,
    "attack_siege": True,
    "attack_support": True,
    "attack_harass": True,
    "retreat_to_base": False,  # Always goes to closest CC
    "final_push": False,       # Always goes to enemy base
}