"""
Terran building construction functions.
Handles all building placement, addons, and upgrades.

MAP AGNOSTIC: No hardcoded map dimensions or positions.
"""

from pysc2.lib import actions, features, units
import numpy as np
from terran_helper import (
    find_best_build_tile, 
    get_base_location,
    is_near_any_command_center
)


# ============================================================================
# CORE BUILDINGS
# ============================================================================

def build_supply_depot(obs, tile_x=None, tile_y=None):
    """Build a Supply Depot at specified or optimal location"""
    # HARD RESTRICTION: Reject coordinates outside radius 8 of any CC
    if tile_x is not None and tile_y is not None:
        if not is_near_any_command_center(obs, tile_x, tile_y, max_radius=16):
            return actions.RAW_FUNCTIONS.no_op()
    
    scvs = [u for u in obs.observation.raw_units 
            if u.unit_type == units.Terran.SCV and u.alliance == features.PlayerRelative.SELF]
    if not scvs:
        return actions.RAW_FUNCTIONS.no_op()
    
    scv = scvs[0]
    
    if tile_x is None or tile_y is None:
        pos = find_best_build_tile(obs, "supply_depot")
        if pos is None:
            return actions.RAW_FUNCTIONS.no_op()
        tile_x, tile_y = pos
    
    return actions.RAW_FUNCTIONS.Build_SupplyDepot_pt("now", scv.tag, (tile_x, tile_y))


def build_command_center(obs, base_tracker):
    """
    Build Command Center at next expansion - MAP AGNOSTIC.
    GATED: Requires current base COMPLETE (1 anchor, 3 production, 5 accessory).
    get_available_expansions() internally checks can_expand() before returning any locations.
    """
    if obs.observation.player.minerals < 400:
        return actions.RAW_FUNCTIONS.no_op()
    if not base_tracker.expansion_centroids:
        base_tracker.initialize_expansions(obs)
    
    # get_available_expansions() will return empty list if can_expand() == False
    available_expansions = base_tracker.get_available_expansions(obs)
    if not available_expansions:
        return actions.RAW_FUNCTIONS.no_op()
    
    for i, centroid in enumerate(available_expansions):
        # Check if already built
        already_built = False
        for u in obs.observation.raw_units:
            if (u.alliance == features.PlayerRelative.SELF and 
                u.unit_type in base_tracker.TIER_1_ANCHOR):
                dist = np.sqrt((u.x - centroid[0])**2 + (u.y - centroid[1])**2)
                if dist < 15:
                    already_built = True
                    break
        if already_built:
            continue
        
        # Find nearby minerals
        minerals = [
            u for u in obs.observation.raw_units
            if u.alliance == features.PlayerRelative.NEUTRAL
            and u.unit_type in (units.Neutral.MineralField, 
                                units.Neutral.MineralField750,
                                units.Neutral.LabMineralField,
                                units.Neutral.LabMineralField750,
                                units.Neutral.RichMineralField,
                                units.Neutral.RichMineralField750)
        ]
        nearby_minerals = [
            m for m in minerals
            if np.sqrt((m.x - centroid[0])**2 + (m.y - centroid[1])**2) < 10
        ]
        if not nearby_minerals:
            continue
        
        mineral_x = sum(m.x for m in nearby_minerals) / len(nearby_minerals)
        mineral_y = sum(m.y for m in nearby_minerals) / len(nearby_minerals)
        
        # Get map center DYNAMICALLY - no hardcoded values
        all_units = obs.observation.raw_units
        if len(all_units) > 0:
            max_x = max(u.x for u in all_units)
            max_y = max(u.y for u in all_units)
            # Estimate map center from unit spread
            map_center_x = max_x / 2 + 10  # Slight offset toward center
            map_center_y = max_y / 2 + 10
        else:
            # Fallback: use centroid position to estimate
            map_center_x = centroid[0]
            map_center_y = centroid[1]
        
        # Direction from minerals toward map center
        dx = map_center_x - mineral_x
        dy = map_center_y - mineral_y
        length = np.sqrt(dx*dx + dy*dy)
        if length > 0:
            dx /= length
            dy /= length
        
        # Place CC offset from minerals toward center
        target_x = int(mineral_x + dx * 8)
        target_y = int(mineral_y + dy * 8)
        
        scvs = [
            u for u in obs.observation.raw_units
            if u.alliance == features.PlayerRelative.SELF
            and u.unit_type == units.Terran.SCV
        ]
        if not scvs:
            return actions.RAW_FUNCTIONS.no_op()
        
        scv = scvs[0]
        return actions.RAW_FUNCTIONS.Build_CommandCenter_pt(
            "now",
            scv.tag,
            [target_x, target_y]
        )
    return actions.RAW_FUNCTIONS.no_op()


def build_barracks(obs, tile_x=None, tile_y=None):
    """Build a Barracks at specified or optimal location"""
    # HARD RESTRICTION: Reject coordinates outside radius 8 of any CC
    if tile_x is not None and tile_y is not None:
        if not is_near_any_command_center(obs, tile_x, tile_y, max_radius=16):
            return actions.RAW_FUNCTIONS.no_op()
    
    scvs = [u for u in obs.observation.raw_units
            if u.alliance == features.PlayerRelative.SELF and
               u.unit_type == units.Terran.SCV]
    if not scvs:
        return actions.RAW_FUNCTIONS.no_op()
    
    scv = scvs[0]
    
    if tile_x is None or tile_y is None:
        pos = find_best_build_tile(obs, "barracks")
        if pos is None:
            return actions.RAW_FUNCTIONS.no_op()
        tile_x, tile_y = pos
    
    return actions.RAW_FUNCTIONS.Build_Barracks_pt("now", scv.tag, (tile_x, tile_y))


def build_factory(obs, tile_x=None, tile_y=None):
    """Build a Factory at specified or optimal location"""
    # HARD RESTRICTION: Reject coordinates outside radius 8 of any CC
    if tile_x is not None and tile_y is not None:
        if not is_near_any_command_center(obs, tile_x, tile_y, max_radius=16):
            return actions.RAW_FUNCTIONS.no_op()
    
    scvs = [u for u in obs.observation.raw_units
            if u.alliance == features.PlayerRelative.SELF and
               u.unit_type == units.Terran.SCV]
    if not scvs:
        return actions.RAW_FUNCTIONS.no_op()
    
    scv = scvs[0]
    
    if tile_x is None or tile_y is None:
        pos = find_best_build_tile(obs, "factory")
        if pos is None:
            return actions.RAW_FUNCTIONS.no_op()
        tile_x, tile_y = pos
    
    return actions.RAW_FUNCTIONS.Build_Factory_pt("now", scv.tag, (tile_x, tile_y))


def build_starport(obs, tile_x=None, tile_y=None):
    """Build a Starport at specified or optimal location"""
    # HARD RESTRICTION: Reject coordinates outside radius 8 of any CC
    if tile_x is not None and tile_y is not None:
        if not is_near_any_command_center(obs, tile_x, tile_y, max_radius=16):
            return actions.RAW_FUNCTIONS.no_op()
    
    scvs = [u for u in obs.observation.raw_units
            if u.alliance == features.PlayerRelative.SELF and
               u.unit_type == units.Terran.SCV]
    if not scvs:
        return actions.RAW_FUNCTIONS.no_op()
    
    scv = scvs[0]
    
    if tile_x is None or tile_y is None:
        pos = find_best_build_tile(obs, "starport")
        if pos is None:
            return actions.RAW_FUNCTIONS.no_op()
        tile_x, tile_y = pos
    
    return actions.RAW_FUNCTIONS.Build_Starport_pt("now", scv.tag, (tile_x, tile_y))


def build_refinery(obs):
    """Build a Refinery on the nearest vespene geyser"""
    scvs = [
        u for u in obs.observation.raw_units
        if u.unit_type == units.Terran.SCV
        and u.alliance == features.PlayerRelative.SELF
    ]
    if not scvs:
        return actions.RAW_FUNCTIONS.no_op()
    
    scv = scvs[0]
    
    geysers = [
        u for u in obs.observation.raw_units
        if u.alliance == features.PlayerRelative.NEUTRAL and
        u.unit_type in (
            units.Neutral.VespeneGeyser,
            units.Neutral.RichVespeneGeyser
        )
    ]
    
    if not geysers:
        return actions.RAW_FUNCTIONS.no_op()
    
    # HARD RESTRICTION: Only build on geysers within radius 8 of a CC
    valid_geysers = [g for g in geysers if is_near_any_command_center(obs, g.x, g.y, max_radius=16)]
    
    if not valid_geysers:
        return actions.RAW_FUNCTIONS.no_op()
    
    closest = min(valid_geysers, key=lambda g: (g.x - scv.x)**2 + (g.y - scv.y)**2)
    
    return actions.RAW_FUNCTIONS.Build_Refinery_pt(
        "now",
        [scv.tag],
        [closest.tag]
    )


# ============================================================================
# TECH BUILDINGS
# ============================================================================

def build_engineering_bay(obs, tile_x=None, tile_y=None):
    """Build an Engineering Bay"""
    # HARD RESTRICTION: Reject coordinates outside radius 8 of any CC
    if tile_x is not None and tile_y is not None:
        if not is_near_any_command_center(obs, tile_x, tile_y, max_radius=16):
            return actions.RAW_FUNCTIONS.no_op()
    
    scvs = [u for u in obs.observation.raw_units
            if u.alliance == features.PlayerRelative.SELF and u.unit_type == units.Terran.SCV]
    if not scvs:
        return actions.RAW_FUNCTIONS.no_op()
    
    scv = scvs[0]
    
    if tile_x is None or tile_y is None:
        pos = find_best_build_tile(obs, "engineering_bay")
        if pos is None:
            return actions.RAW_FUNCTIONS.no_op()
        tile_x, tile_y = pos
    
    return actions.RAW_FUNCTIONS.Build_EngineeringBay_pt("now", scv.tag, (tile_x, tile_y))


def build_armory(obs, tile_x=None, tile_y=None):
    """Build an Armory"""
    # HARD RESTRICTION: Reject coordinates outside radius 8 of any CC
    if tile_x is not None and tile_y is not None:
        if not is_near_any_command_center(obs, tile_x, tile_y, max_radius=16):
            return actions.RAW_FUNCTIONS.no_op()
    
    scvs = [u for u in obs.observation.raw_units
            if u.alliance == features.PlayerRelative.SELF and u.unit_type == units.Terran.SCV]
    if not scvs:
        return actions.RAW_FUNCTIONS.no_op()
    
    scv = scvs[0]
    
    if tile_x is None or tile_y is None:
        pos = find_best_build_tile(obs, "armory")
        if pos is None:
            return actions.RAW_FUNCTIONS.no_op()
        tile_x, tile_y = pos
    
    return actions.RAW_FUNCTIONS.Build_Armory_pt("now", scv.tag, (tile_x, tile_y))


def build_fusion_core(obs, tile_x=None, tile_y=None):
    """Build a Fusion Core"""
    # HARD RESTRICTION: Reject coordinates outside radius 8 of any CC
    if tile_x is not None and tile_y is not None:
        if not is_near_any_command_center(obs, tile_x, tile_y, max_radius=16):
            return actions.RAW_FUNCTIONS.no_op()
    
    scvs = [u for u in obs.observation.raw_units
            if u.alliance == features.PlayerRelative.SELF and u.unit_type == units.Terran.SCV]
    if not scvs:
        return actions.RAW_FUNCTIONS.no_op()
    
    scv = scvs[0]
    
    if tile_x is None or tile_y is None:
        pos = find_best_build_tile(obs, "fusion_core")
        if pos is None:
            return actions.RAW_FUNCTIONS.no_op()
        tile_x, tile_y = pos
    
    return actions.RAW_FUNCTIONS.Build_FusionCore_pt("now", scv.tag, (tile_x, tile_y))


def build_ghost_academy(obs, tile_x=None, tile_y=None):
    """Build a Ghost Academy"""
    # HARD RESTRICTION: Reject coordinates outside radius 8 of any CC
    if tile_x is not None and tile_y is not None:
        if not is_near_any_command_center(obs, tile_x, tile_y, max_radius=16):
            return actions.RAW_FUNCTIONS.no_op()
    
    scvs = [u for u in obs.observation.raw_units
            if u.alliance == features.PlayerRelative.SELF and u.unit_type == units.Terran.SCV]
    if not scvs:
        return actions.RAW_FUNCTIONS.no_op()
    
    scv = scvs[0]
    
    if tile_x is None or tile_y is None:
        pos = find_best_build_tile(obs, "ghost_academy")
        if pos is None:
            return actions.RAW_FUNCTIONS.no_op()
        tile_x, tile_y = pos
    
    return actions.RAW_FUNCTIONS.Build_GhostAcademy_pt("now", scv.tag, (tile_x, tile_y))


# ============================================================================
# DEFENSIVE BUILDINGS
# ============================================================================

def build_bunker(obs, tile_x=None, tile_y=None):
    """Build a Bunker"""
    # HARD RESTRICTION: Reject coordinates outside radius 8 of any CC
    if tile_x is not None and tile_y is not None:
        if not is_near_any_command_center(obs, tile_x, tile_y, max_radius=16):
            return actions.RAW_FUNCTIONS.no_op()
    
    scvs = [u for u in obs.observation.raw_units
            if u.alliance == features.PlayerRelative.SELF and u.unit_type == units.Terran.SCV]
    if not scvs:
        return actions.RAW_FUNCTIONS.no_op()
    
    scv = scvs[0]
    
    if tile_x is None or tile_y is None:
        pos = find_best_build_tile(obs, "bunker")
        if pos is None:
            return actions.RAW_FUNCTIONS.no_op()
        tile_x, tile_y = pos
    
    return actions.RAW_FUNCTIONS.Build_Bunker_pt("now", scv.tag, (tile_x, tile_y))


def build_missile_turret(obs, tile_x, tile_y):
    """Build a Missile Turret at specified location"""
    # HARD RESTRICTION: Reject coordinates outside radius 8 of any CC
    if tile_x is not None and tile_y is not None:
        if not is_near_any_command_center(obs, tile_x, tile_y, max_radius=16):
            return actions.RAW_FUNCTIONS.no_op()
    
    scvs = [u for u in obs.observation.raw_units
            if u.alliance == features.PlayerRelative.SELF and u.unit_type == units.Terran.SCV]
    if not scvs:
        return actions.RAW_FUNCTIONS.no_op()
    
    scv = scvs[0]
    return actions.RAW_FUNCTIONS.Build_MissileTurret_pt("now", scv.tag, (tile_x, tile_y))


def build_sensor_tower(obs, tile_x, tile_y):
    """Build a Sensor Tower at specified location"""
    # HARD RESTRICTION: Reject coordinates outside radius 8 of any CC
    if tile_x is not None and tile_y is not None:
        if not is_near_any_command_center(obs, tile_x, tile_y, max_radius=16):
            return actions.RAW_FUNCTIONS.no_op()
    
    scvs = [u for u in obs.observation.raw_units
            if u.alliance == features.PlayerRelative.SELF and u.unit_type == units.Terran.SCV]
    if not scvs:
        return actions.RAW_FUNCTIONS.no_op()
    
    scv = scvs[0]
    return actions.RAW_FUNCTIONS.Build_SensorTower_pt("now", scv.tag, (tile_x, tile_y))


# ============================================================================
# ADDONS - Individual functions for each type
# ============================================================================

def add_barracks_techlab(obs, _x=None, _y=None):
    """Add Tech Lab to Barracks"""
    barracks_ready = [
        b for b in obs.observation.raw_units
        if b.alliance == features.PlayerRelative.SELF
        and b.unit_type == units.Terran.Barracks
        and b.build_progress == 100
        and getattr(b, "add_on_tag", 0) == 0
    ]
    if not barracks_ready:
        return actions.RAW_FUNCTIONS.no_op()
    
    barracks = barracks_ready[0]
    return actions.RAW_FUNCTIONS.Build_TechLab_Barracks_quick("now", barracks.tag)


def add_barracks_reactor(obs, _x=None, _y=None):
    """Add Reactor to Barracks"""
    barracks_ready = [
        b for b in obs.observation.raw_units
        if b.alliance == features.PlayerRelative.SELF
        and b.unit_type == units.Terran.Barracks
        and b.build_progress == 100
        and getattr(b, "add_on_tag", 0) == 0
    ]
    if not barracks_ready:
        return actions.RAW_FUNCTIONS.no_op()
    
    barracks = barracks_ready[0]
    return actions.RAW_FUNCTIONS.Build_Reactor_Barracks_quick("now", barracks.tag)


def add_factory_techlab(obs, _x=None, _y=None):
    """Add Tech Lab to Factory"""
    factories_ready = [
        f for f in obs.observation.raw_units
        if f.alliance == features.PlayerRelative.SELF
        and f.unit_type == units.Terran.Factory
        and f.build_progress == 100
        and getattr(f, "add_on_tag", 0) == 0
    ]
    if not factories_ready:
        return actions.RAW_FUNCTIONS.no_op()
    
    factory = factories_ready[0]
    return actions.RAW_FUNCTIONS.Build_TechLab_Factory_quick("now", factory.tag)


def add_factory_reactor(obs, _x=None, _y=None):
    """Add Reactor to Factory"""
    factories_ready = [
        f for f in obs.observation.raw_units
        if f.alliance == features.PlayerRelative.SELF
        and f.unit_type == units.Terran.Factory
        and f.build_progress == 100
        and getattr(f, "add_on_tag", 0) == 0
    ]
    if not factories_ready:
        return actions.RAW_FUNCTIONS.no_op()
    
    factory = factories_ready[0]
    return actions.RAW_FUNCTIONS.Build_Reactor_Factory_quick("now", factory.tag)


def add_starport_techlab(obs, _x=None, _y=None):
    """Add Tech Lab to Starport"""
    starports_ready = [
        s for s in obs.observation.raw_units
        if s.alliance == features.PlayerRelative.SELF
        and s.unit_type == units.Terran.Starport
        and s.build_progress == 100
        and getattr(s, "add_on_tag", 0) == 0
    ]
    if not starports_ready:
        return actions.RAW_FUNCTIONS.no_op()
    
    starport = starports_ready[0]
    return actions.RAW_FUNCTIONS.Build_TechLab_Starport_quick("now", starport.tag)


def add_starport_reactor(obs, _x=None, _y=None):
    """Add Reactor to Starport"""
    starports_ready = [
        s for s in obs.observation.raw_units
        if s.alliance == features.PlayerRelative.SELF
        and s.unit_type == units.Terran.Starport
        and s.build_progress == 100
        and getattr(s, "add_on_tag", 0) == 0
    ]
    if not starports_ready:
        return actions.RAW_FUNCTIONS.no_op()
    
    starport = starports_ready[0]
    return actions.RAW_FUNCTIONS.Build_Reactor_Starport_quick("now", starport.tag)


# Legacy aliases for backward compatibility
def add_barracks_addon(obs, _x=None, _y=None):
    return add_barracks_techlab(obs, _x, _y)

def add_factory_addon(obs, _x=None, _y=None):
    return add_factory_techlab(obs, _x, _y)

def add_starport_addon(obs, _x=None, _y=None):
    return add_starport_techlab(obs, _x, _y)

def build_techlab(obs, _x=None, _y=None):
    return add_factory_techlab(obs, _x, _y)

def build_reactor(obs, _x=None, _y=None):
    return add_factory_reactor(obs, _x, _y)


# ============================================================================
# BUILDING UPGRADES
# ============================================================================

def upgrade_orbital_command(obs, _x=None, _y=None):
    """Upgrade Command Center to Orbital Command"""
    ccs = [u for u in obs.observation.raw_units
           if u.alliance == features.PlayerRelative.SELF 
           and u.unit_type == units.Terran.CommandCenter
           and u.build_progress == 100]
    if not ccs:
        return actions.RAW_FUNCTIONS.no_op()
    
    cc = ccs[0]
    return actions.RAW_FUNCTIONS.Morph_OrbitalCommand_quick("now", cc.tag)


def upgrade_planetary_fortress(obs, _x=None, _y=None):
    """Upgrade Command Center to Planetary Fortress"""
    ccs = [u for u in obs.observation.raw_units
           if u.alliance == features.PlayerRelative.SELF 
           and u.unit_type == units.Terran.CommandCenter
           and u.build_progress == 100]
    if not ccs:
        return actions.RAW_FUNCTIONS.no_op()
    
    cc = ccs[0]
    return actions.RAW_FUNCTIONS.Morph_PlanetaryFortress_quick("now", cc.tag)


def lower_depot(obs, _x=None, _y=None):
    """Lower a Supply Depot"""
    depots = [
        u for u in obs.observation.raw_units
        if u.alliance == features.PlayerRelative.SELF and
           u.unit_type == units.Terran.SupplyDepot
    ]
    if not depots:
        return actions.RAW_FUNCTIONS.no_op()
    
    depot = depots[0]
    return actions.RAW_FUNCTIONS.Morph_SupplyDepot_Lower_quick("now", depot.tag)