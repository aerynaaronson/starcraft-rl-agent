"""
Helper functions for Terran bot operations.
Includes utility functions for distance calculations, positioning, and map analysis.
"""

from pysc2.lib import actions, features, units
from math import sqrt


def distance(a, b):
    """Calculate Euclidean distance between two points"""
    return sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


def get_base_location(obs):
    """Get the position of the main base (first Command Center)"""
    ccs = [u for u in obs.observation.raw_units
           if u.alliance == features.PlayerRelative.SELF 
           and u.unit_type in [units.Terran.CommandCenter, 
                              units.Terran.OrbitalCommand,
                              units.Terran.PlanetaryFortress]]
    if ccs:
        return (int(ccs[0].x), int(ccs[0].y))
    
    structures = [u for u in obs.observation.raw_units
                 if u.alliance == features.PlayerRelative.SELF and u.is_structure]
    if structures:
        return (int(structures[0].x), int(structures[0].y))
    
    return (50, 50)


def get_all_command_centers(obs):
    """Get positions of all Command Centers (CC, Orbital, Planetary)"""
    ccs = [u for u in obs.observation.raw_units
           if u.alliance == features.PlayerRelative.SELF 
           and u.unit_type in [units.Terran.CommandCenter, 
                              units.Terran.OrbitalCommand,
                              units.Terran.PlanetaryFortress]]
    return [(int(cc.x), int(cc.y)) for cc in ccs]


def get_all_structures(obs):
    """Get positions of all friendly structures"""
    return [(int(u.x), int(u.y)) for u in obs.observation.raw_units
            if u.alliance == features.PlayerRelative.SELF and u.is_structure]


def is_position_blocked(obs, x, y, min_distance=5):
    """Check if a position is too close to existing structures"""
    structures = get_all_structures(obs)
    for sx, sy in structures:
        if distance((x, y), (sx, sy)) < min_distance:
            return True
    return False


def is_near_any_command_center(obs, x, y, max_radius=8):
    """
    HARD RESTRICTION: Check if position is within max_radius of ANY Command Center.
    Returns False if position is outside radius 8 of all CCs.
    """
    ccs = get_all_command_centers(obs)
    if not ccs:
        # If no CCs exist yet, don't allow building (shouldn't happen)
        return False
    
    for cc_x, cc_y in ccs:
        if distance((x, y), (cc_x, cc_y)) <= max_radius:
            return True
    return False


def get_map_bounds(obs):
    """Get approximate map dimensions"""
    try:
        all_units = obs.observation.raw_units
        if all_units:
            max_x = max(u.x for u in all_units)
            max_y = max(u.y for u in all_units)
            return int(max_x + 20), int(max_y + 20)
    except:
        pass
    return 200, 200


def find_best_build_tile(obs, structure_type, preferred_center=None, max_radius=8):
    """
    HARD RESTRICTION: Find placement ONLY within radius 8 of any Command Center.
    
    Args:
        obs: Game observation
        structure_type: Type of structure
        preferred_center: (x, y) preferred center, defaults to nearest CC with space
        max_radius: Maximum search radius (HARD CAP at 8)
        
    Returns:
        (x, y) position or None if no valid position found
    """
    # Get all command centers
    ccs = get_all_command_centers(obs)
    if not ccs:
        return None  # No CCs = no building allowed
    
    # HARD CAP max_radius at 8
    max_radius = min(max_radius, 8)
    
    # If no preferred center, find CC with most space available
    if preferred_center is None:
        structures = get_all_structures(obs)
        cc_space_scores = []
        for cc in ccs:
            nearby_structures = sum(1 for s in structures if distance(cc, s) <= 8)
            # Prefer CCs with fewer structures (more space)
            cc_space_scores.append((nearby_structures, cc))
        cc_space_scores.sort()
        preferred_center = cc_space_scores[0][1]
    
    cx, cy = preferred_center
    map_w, map_h = get_map_bounds(obs)
    
    # Structure-specific preferences
    STRUCTURE_PREFS = {
        "supply_depot": {"ideal_dist": 6, "min_spacing": 2},
        "barracks": {"ideal_dist": 7, "min_spacing": 3},
        "factory": {"ideal_dist": 7, "min_spacing": 3},
        "starport": {"ideal_dist": 7, "min_spacing": 3},
        "engineering_bay": {"ideal_dist": 6, "min_spacing": 3},
        "armory": {"ideal_dist": 6, "min_spacing": 3},
        "fusion_core": {"ideal_dist": 7, "min_spacing": 3},
        "ghost_academy": {"ideal_dist": 7, "min_spacing": 3},
        "missile_turret": {"ideal_dist": 7, "min_spacing": 4},
        "bunker": {"ideal_dist": 7, "min_spacing": 3},
        "sensor_tower": {"ideal_dist": 7, "min_spacing": 5},
    }
    
    prefs = STRUCTURE_PREFS.get(structure_type, {"ideal_dist": 6, "min_spacing": 3})
    ideal_dist = prefs["ideal_dist"]
    min_spacing = prefs["min_spacing"]
    
    best_tile = None
    best_score = float('inf')
    candidates = []
    
    # Generate candidate positions in tight spiral
    for radius in range(2, max_radius + 1, 1):
        for angle in range(0, 360, 20):
            import math
            dx = int(radius * math.cos(math.radians(angle)))
            dy = int(radius * math.sin(math.radians(angle)))
            x, y = cx + dx, cy + dy
            
            # Check map bounds
            if x < 5 or x >= map_w - 5 or y < 5 or y >= map_h - 5:
                continue
            
            # HARD RESTRICTION: Must be within radius 8 of ANY CC
            if not is_near_any_command_center(obs, x, y, max_radius=16):
                continue
            
            # Check if position is blocked
            if is_position_blocked(obs, x, y, min_spacing):
                continue
            
            candidates.append((x, y))
    
    # Score candidates
    for x, y in candidates:
        dist_to_center = distance((x, y), preferred_center)
        score = abs(dist_to_center - ideal_dist)
        
        # Prefer positions with good spacing
        same_type_structures = get_all_structures(obs)
        if same_type_structures:
            min_dist_to_same = min(distance((x, y), s) for s in same_type_structures)
            score -= min_dist_to_same * 0.1
        
        if score < best_score:
            best_score = score
            best_tile = (x, y)
    
    return best_tile


def get_expansion_count(obs):
    """Get the current number of bases"""
    bases = [
        u for u in obs.observation.raw_units
        if u.alliance == features.PlayerRelative.SELF
        and u.unit_type in [
            units.Terran.CommandCenter,
            units.Terran.OrbitalCommand,
            units.Terran.PlanetaryFortress,
            units.Terran.CommandCenterFlying,
        ]
    ]
    return len(bases)


def is_air_unit(unit):
    """Return True if the given unit is an air unit."""
    AIR_UNITS = {
        units.Terran.Medivac if hasattr(units.Terran, 'Medivac') else 84,
        units.Terran.VikingFighter if hasattr(units.Terran, 'VikingFighter') else 50,
        units.Terran.VikingAssault if hasattr(units.Terran, 'VikingAssault') else 49,
        units.Terran.Liberator if hasattr(units.Terran, 'Liberator') else 67,
        units.Terran.LiberatorAG if hasattr(units.Terran, 'LiberatorAG') else 68,
        units.Terran.Raven if hasattr(units.Terran, 'Raven') else 59,
        units.Terran.Banshee if hasattr(units.Terran, 'Banshee') else 48,
        units.Terran.Battlecruiser if hasattr(units.Terran, 'Battlecruiser') else 11,
    }
    return unit.unit_type in AIR_UNITS


def is_ground_unit(unit):
    """Return True if the given unit is a ground unit."""
    return not is_air_unit(unit)