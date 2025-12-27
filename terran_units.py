"""
Terran unit training, abilities, and research.
Handles all unit-related actions.
"""

from pysc2.lib import actions, features, units


# ============================================================================
# WORKER MANAGEMENT
# ============================================================================

def train_worker(obs, _x=None, _y=None):
    """Train an SCV from Command Center"""
    townhalls = [
        u for u in obs.observation.raw_units
        if u.alliance == features.PlayerRelative.SELF and
           u.unit_type == units.Terran.CommandCenter
    ]
    if not townhalls:
        return actions.RAW_FUNCTIONS.no_op()
    
    cc = townhalls[0]
    return actions.RAW_FUNCTIONS.Train_SCV_quick("now", cc.tag)


def send_idle_workers_to_mine(obs, _x=None, _y=None):
    """Send idle SCVs to mine minerals"""
    idle_scvs = [u for u in obs.observation.raw_units 
                 if u.unit_type == units.Terran.SCV 
                 and u.alliance == features.PlayerRelative.SELF
                 and u.order_length == 0]
    
    if not idle_scvs:
        return actions.RAW_FUNCTIONS.no_op()
    
    mineral_patches = [u for u in obs.observation.raw_units 
                      if u.alliance == features.PlayerRelative.NEUTRAL and
                      u.unit_type in [units.Neutral.MineralField, 
                                     units.Neutral.MineralField750,
                                     units.Neutral.RichMineralField,
                                     units.Neutral.RichMineralField750]]
    
    if not mineral_patches:
        return actions.RAW_FUNCTIONS.no_op()
    
    scv = idle_scvs[0]
    min_dist = float('inf')
    closest_mineral = mineral_patches[0]
    
    for m in mineral_patches:
        dist = ((m.x - scv.x)**2 + (m.y - scv.y)**2)**0.5
        if dist < min_dist:
            min_dist = dist
            closest_mineral = m
    
    return actions.RAW_FUNCTIONS.Harvest_Gather_SCV_unit("now", scv.tag, closest_mineral.tag)


def send_idle_workers_to_gas(obs, _x=None, _y=None):
    """Send idle SCVs to harvest vespene gas"""
    refineries = [
        u for u in obs.observation.raw_units
        if u.alliance == features.PlayerRelative.SELF
        and u.unit_type == units.Terran.Refinery
    ]
    
    def refinery_load(ref, obs):
        return sum(
            1 for u in obs.observation.raw_units
            if u.alliance == features.PlayerRelative.SELF
            and u.unit_type == units.Terran.SCV
            and getattr(u, "order_id_0", -1) == actions.RAW_FUNCTIONS.Harvest_Gather_unit.id
            and getattr(u, "order_target", None) == ref.tag
        )
    
    candidates = [ref for ref in refineries if refinery_load(ref, obs) < 3]
    
    if not candidates:
        return actions.RAW_FUNCTIONS.no_op()
    
    scvs = [
        u for u in obs.observation.raw_units
        if u.unit_type == units.Terran.SCV
        and u.alliance == features.PlayerRelative.SELF
        and getattr(u, "order_length", 1) == 0
    ]
    
    if not scvs:
        return actions.RAW_FUNCTIONS.no_op()
    
    scv = scvs[0]
    refinery = candidates[0]
    
    return actions.RAW_FUNCTIONS.Harvest_Gather_unit("now", scv.tag, refinery.tag)


# ============================================================================
# BARRACKS UNITS
# ============================================================================

def train_marine(obs, _x=None, _y=None):
    """Train a Marine from Barracks"""
    barracks = [
        u for u in obs.observation.raw_units
        if u.alliance == features.PlayerRelative.SELF and
           u.unit_type == units.Terran.Barracks
    ]
    if not barracks:
        return actions.RAW_FUNCTIONS.no_op()
    
    rax = barracks[0]
    return actions.RAW_FUNCTIONS.Train_Marine_quick("now", rax.tag)


def train_marauder(obs, _x=None, _y=None):
    """Train a Marauder from Barracks (requires Tech Lab)"""
    barracks = [
        u for u in obs.observation.raw_units
        if u.alliance == features.PlayerRelative.SELF and
           u.unit_type == units.Terran.Barracks
    ]
    if not barracks:
        return actions.RAW_FUNCTIONS.no_op()
    
    rax = barracks[0]
    return actions.RAW_FUNCTIONS.Train_Marauder_quick("now", rax.tag)


def train_reaper(obs, _x=None, _y=None):
    """Train a Reaper from Barracks"""
    barracks = [
        u for u in obs.observation.raw_units
        if u.alliance == features.PlayerRelative.SELF and
           u.unit_type == units.Terran.Barracks
    ]
    if not barracks:
        return actions.RAW_FUNCTIONS.no_op()
    
    rax = barracks[0]
    return actions.RAW_FUNCTIONS.Train_Reaper_quick("now", rax.tag)


def train_ghost(obs, _x=None, _y=None):
    """Train a Ghost from Barracks (requires Ghost Academy)"""
    barracks = [u for u in obs.observation.raw_units
                 if u.alliance == features.PlayerRelative.SELF
                 and u.unit_type == units.Terran.Barracks]
    if not barracks:
        return actions.RAW_FUNCTIONS.no_op()
    
    rax = barracks[0]
    return actions.RAW_FUNCTIONS.Train_Ghost_quick("now", rax.tag)


# ============================================================================
# FACTORY UNITS
# ============================================================================

def train_hellion(obs, _x=None, _y=None):
    """Train a Hellion from Factory"""
    factories = [u for u in obs.observation.raw_units
                 if u.alliance == features.PlayerRelative.SELF
                 and u.unit_type == units.Terran.Factory]
    if not factories:
        return actions.RAW_FUNCTIONS.no_op()
    
    factory = factories[0]
    return actions.RAW_FUNCTIONS.Train_Hellion_quick("now", factory.tag)


def train_hellbat(obs, _x=None, _y=None):
    """Train a Hellbat from Factory (trains as Hellion, then transforms)"""
    factories = [u for u in obs.observation.raw_units
                 if u.alliance == features.PlayerRelative.SELF
                 and u.unit_type == units.Terran.Factory]
    if not factories:
        return actions.RAW_FUNCTIONS.no_op()
    
    factory = factories[0]
    return actions.RAW_FUNCTIONS.Train_Hellion_quick("now", factory.tag)


def train_siege_tank(obs, _x=None, _y=None):
    """Train a Siege Tank from Factory (requires Tech Lab)"""
    factories = [u for u in obs.observation.raw_units
                 if u.alliance == features.PlayerRelative.SELF
                 and u.unit_type == units.Terran.Factory]
    if not factories:
        return actions.RAW_FUNCTIONS.no_op()
    
    factory = factories[0]
    return actions.RAW_FUNCTIONS.Train_SiegeTank_quick("now", factory.tag)


def train_thor(obs, _x=None, _y=None):
    """Train a Thor from Factory (requires Tech Lab and Armory)"""
    factories = [u for u in obs.observation.raw_units
                 if u.alliance == features.PlayerRelative.SELF
                 and u.unit_type == units.Terran.Factory]
    if not factories:
        return actions.RAW_FUNCTIONS.no_op()
    
    factory = factories[0]
    return actions.RAW_FUNCTIONS.Train_Thor_quick("now", factory.tag)


def train_cyclone(obs, _x=None, _y=None):
    """Train a Cyclone from Factory (requires Tech Lab)"""
    factories = [u for u in obs.observation.raw_units
                 if u.alliance == features.PlayerRelative.SELF
                 and u.unit_type == units.Terran.Factory]
    if not factories:
        return actions.RAW_FUNCTIONS.no_op()
    
    factory = factories[0]
    return actions.RAW_FUNCTIONS.Train_Cyclone_quick("now", factory.tag)


def train_widow_mine(obs, _x=None, _y=None):
    """Train a Widow Mine from Factory"""
    factories = [u for u in obs.observation.raw_units
                 if u.alliance == features.PlayerRelative.SELF
                 and u.unit_type == units.Terran.Factory]
    if not factories:
        return actions.RAW_FUNCTIONS.no_op()
    
    factory = factories[0]
    return actions.RAW_FUNCTIONS.Train_WidowMine_quick("now", factory.tag)


# ============================================================================
# STARPORT UNITS
# ============================================================================

def train_viking(obs, _x=None, _y=None):
    """Train a Viking from Starport"""
    starports = [u for u in obs.observation.raw_units
                 if u.alliance == features.PlayerRelative.SELF
                 and u.unit_type == units.Terran.Starport]
    if not starports:
        return actions.RAW_FUNCTIONS.no_op()
    
    starport = starports[0]
    return actions.RAW_FUNCTIONS.Train_VikingFighter_quick("now", starport.tag)


def train_medivac(obs, _x=None, _y=None):
    """Train a Medivac from Starport"""
    starports = [u for u in obs.observation.raw_units
                 if u.alliance == features.PlayerRelative.SELF
                 and u.unit_type == units.Terran.Starport]
    if not starports:
        return actions.RAW_FUNCTIONS.no_op()
    
    starport = starports[0]
    return actions.RAW_FUNCTIONS.Train_Medivac_quick("now", starport.tag)


def train_liberator(obs, _x=None, _y=None):
    """Train a Liberator from Starport"""
    starports = [u for u in obs.observation.raw_units
                 if u.alliance == features.PlayerRelative.SELF
                 and u.unit_type == units.Terran.Starport]
    if not starports:
        return actions.RAW_FUNCTIONS.no_op()
    
    starport = starports[0]
    return actions.RAW_FUNCTIONS.Train_Liberator_quick("now", starport.tag)


def train_raven(obs, _x=None, _y=None):
    """Train a Raven from Starport (requires Tech Lab)"""
    starports = [u for u in obs.observation.raw_units
                 if u.alliance == features.PlayerRelative.SELF
                 and u.unit_type == units.Terran.Starport]
    if not starports:
        return actions.RAW_FUNCTIONS.no_op()
    
    starport = starports[0]
    return actions.RAW_FUNCTIONS.Train_Raven_quick("now", starport.tag)


def train_banshee(obs, _x=None, _y=None):
    """Train a Banshee from Starport (requires Tech Lab)"""
    starports = [u for u in obs.observation.raw_units
                 if u.alliance == features.PlayerRelative.SELF
                 and u.unit_type == units.Terran.Starport]
    if not starports:
        return actions.RAW_FUNCTIONS.no_op()
    
    starport = starports[0]
    return actions.RAW_FUNCTIONS.Train_Banshee_quick("now", starport.tag)


def train_battlecruiser(obs, _x=None, _y=None):
    """Train a Battlecruiser from Starport (requires Tech Lab and Fusion Core)"""
    starports = [u for u in obs.observation.raw_units
                 if u.alliance == features.PlayerRelative.SELF
                 and u.unit_type == units.Terran.Starport]
    if not starports:
        return actions.RAW_FUNCTIONS.no_op()
    
    starport = starports[0]
    return actions.RAW_FUNCTIONS.Train_Battlecruiser_quick("now", starport.tag)


# ============================================================================
# ORBITAL COMMAND ABILITIES
# ============================================================================

def call_down_mule(obs, tile_x, tile_y):
    """Call down a MULE to harvest minerals"""
    orbitals = [u for u in obs.observation.raw_units
                if u.alliance == features.PlayerRelative.SELF 
                and u.unit_type == units.Terran.OrbitalCommand
                and u.energy >= 50]
    if not orbitals:
        return actions.RAW_FUNCTIONS.no_op()
    
    orbital = orbitals[0]
    
    mineral_patches = [u for u in obs.observation.raw_units
                      if u.alliance == features.PlayerRelative.NEUTRAL
                      and u.unit_type in [units.Neutral.MineralField, 
                                         units.Neutral.MineralField750,
                                         units.Neutral.RichMineralField,
                                         units.Neutral.RichMineralField750]]
    if not mineral_patches:
        return actions.RAW_FUNCTIONS.no_op()
    
    closest = min(mineral_patches, 
                 key=lambda m: (m.x - orbital.x)**2 + (m.y - orbital.y)**2)
    
    return actions.RAW_FUNCTIONS.Effect_CalldownMULE_unit("now", orbital.tag, closest.tag)


def scanner_sweep(obs, tile_x, tile_y):
    """Use Scanner Sweep ability"""
    orbitals = [u for u in obs.observation.raw_units
                if u.alliance == features.PlayerRelative.SELF
                and u.unit_type == units.Terran.OrbitalCommand
                and u.energy >= 50]
    if not orbitals:
        return actions.RAW_FUNCTIONS.no_op()
    
    orbital = orbitals[0]
    return actions.RAW_FUNCTIONS.Effect_Scan_pt("now", orbital.tag, (tile_x, tile_y))


# ============================================================================
# RESEARCH
# ============================================================================

# Track research progress for each building type
_research_state = {
    'barracks_techlab': 0,
    'factory_techlab': 0,
    'starport_techlab': 0,
    'engineering_bay': 0,
    'armory': 0,
    'ghost_academy': 0
}


def research_barracks_techlab(obs, _x=None, _y=None):
    """Research next available upgrade at Barracks Tech Lab
    Cycles through: Stimpack -> Combat Shield -> Concussive Shells
    """
    barracks_techlabs = [u for u in obs.observation.raw_units
                         if u.alliance == features.PlayerRelative.SELF
                         and u.unit_type == units.Terran.BarracksTechLab]
    if not barracks_techlabs:
        return actions.RAW_FUNCTIONS.no_op()
    
    techlab = barracks_techlabs[0]
    state = _research_state['barracks_techlab'] % 3
    
    if state == 0:
        _research_state['barracks_techlab'] += 1
        return actions.RAW_FUNCTIONS.Research_Stimpack_quick("now", techlab.tag)
    elif state == 1:
        _research_state['barracks_techlab'] += 1
        return actions.RAW_FUNCTIONS.Research_CombatShield_quick("now", techlab.tag)
    else:
        _research_state['barracks_techlab'] += 1
        return actions.RAW_FUNCTIONS.Research_ConcussiveShells_quick("now", techlab.tag)


def research_factory_techlab(obs, _x=None, _y=None):
    """Research next available upgrade at Factory Tech Lab
    Cycles through: Smart Servos -> Drilling Claws -> Infernal Pre-Igniter -> 
                   Cyclone Rapid Fire -> Cyclone Lock-On Damage
    """
    factory_techlabs = [u for u in obs.observation.raw_units
                        if u.alliance == features.PlayerRelative.SELF
                        and u.unit_type == units.Terran.FactoryTechLab]
    if not factory_techlabs:
        return actions.RAW_FUNCTIONS.no_op()
    
    techlab = factory_techlabs[0]
    state = _research_state['factory_techlab'] % 5
    
    if state == 0:
        _research_state['factory_techlab'] += 1
        return actions.RAW_FUNCTIONS.Research_SmartServos_quick("now", techlab.tag)
    elif state == 1:
        _research_state['factory_techlab'] += 1
        return actions.RAW_FUNCTIONS.Research_DrillingClaws_quick("now", techlab.tag)
    elif state == 2:
        _research_state['factory_techlab'] += 1
        return actions.RAW_FUNCTIONS.Research_InfernalPreigniter_quick("now", techlab.tag)
    elif state == 3:
        _research_state['factory_techlab'] += 1
        return actions.RAW_FUNCTIONS.Research_CycloneRapidFireLaunchers_quick("now", techlab.tag)
    else:
        _research_state['factory_techlab'] += 1
        return actions.RAW_FUNCTIONS.Research_CycloneLockOnDamage_quick("now", techlab.tag)


def research_starport_techlab(obs, _x=None, _y=None):
    """Research next available upgrade at Starport Tech Lab
    Cycles through: Banshee Cloak -> Banshee Speed -> Raven Reactor -> 
                   Raven Explosives -> Battlecruiser Refit
    """
    starport_techlabs = [u for u in obs.observation.raw_units
                         if u.alliance == features.PlayerRelative.SELF
                         and u.unit_type == units.Terran.StarportTechLab]
    if not starport_techlabs:
        return actions.RAW_FUNCTIONS.no_op()
    
    techlab = starport_techlabs[0]
    state = _research_state['starport_techlab'] % 5
    
    if state == 0:
        _research_state['starport_techlab'] += 1
        return actions.RAW_FUNCTIONS.Research_BansheeCloakingField_quick("now", techlab.tag)
    elif state == 1:
        _research_state['starport_techlab'] += 1
        return actions.RAW_FUNCTIONS.Research_BansheeHyperflightRotors_quick("now", techlab.tag)
    elif state == 2:
        _research_state['starport_techlab'] += 1
        return actions.RAW_FUNCTIONS.Research_RavenCorvidReactor_quick("now", techlab.tag)
    elif state == 3:
        _research_state['starport_techlab'] += 1
        return actions.RAW_FUNCTIONS.Research_RavenRecalibratedExplosives_quick("now", techlab.tag)
    else:
        _research_state['starport_techlab'] += 1
        return actions.RAW_FUNCTIONS.Research_BattlecruiserWeaponRefit_quick("now", techlab.tag)


def research_engineering_bay(obs, _x=None, _y=None):
    """Research next available upgrade at Engineering Bay
    Cycles through: Infantry Weapons 1 -> Infantry Armor 1 -> Hi-Sec Tracking -> 
                   Neosteel Frame -> Infantry Weapons 2 -> Infantry Armor 2 -> 
                   Infantry Weapons 3 -> Infantry Armor 3
    """
    eng_bays = [u for u in obs.observation.raw_units
                if u.alliance == features.PlayerRelative.SELF
                and u.unit_type == units.Terran.EngineeringBay]
    if not eng_bays:
        return actions.RAW_FUNCTIONS.no_op()
    
    bay = eng_bays[0]
    state = _research_state['engineering_bay'] % 8
    
    if state == 0:
        _research_state['engineering_bay'] += 1
        return actions.RAW_FUNCTIONS.Research_TerranInfantryWeaponsLevel1_quick("now", bay.tag)
    elif state == 1:
        _research_state['engineering_bay'] += 1
        return actions.RAW_FUNCTIONS.Research_TerranInfantryArmorLevel1_quick("now", bay.tag)
    elif state == 2:
        _research_state['engineering_bay'] += 1
        return actions.RAW_FUNCTIONS.Research_HiSecAutoTracking_quick("now", bay.tag)
    elif state == 3:
        _research_state['engineering_bay'] += 1
        return actions.RAW_FUNCTIONS.Research_NeosteelFrame_quick("now", bay.tag)
    elif state == 4:
        _research_state['engineering_bay'] += 1
        return actions.RAW_FUNCTIONS.Research_TerranInfantryWeaponsLevel2_quick("now", bay.tag)
    elif state == 5:
        _research_state['engineering_bay'] += 1
        return actions.RAW_FUNCTIONS.Research_TerranInfantryArmorLevel2_quick("now", bay.tag)
    elif state == 6:
        _research_state['engineering_bay'] += 1
        return actions.RAW_FUNCTIONS.Research_TerranInfantryWeaponsLevel3_quick("now", bay.tag)
    else:
        _research_state['engineering_bay'] += 1
        return actions.RAW_FUNCTIONS.Research_TerranInfantryArmorLevel3_quick("now", bay.tag)


def research_armory(obs, _x=None, _y=None):
    """Research next available upgrade at Armory
    Cycles through: Vehicle Weapons 1 -> Vehicle Plating 1 -> Ship Weapons 1 -> 
                   Vehicle Weapons 2 -> Vehicle Plating 2 -> Ship Weapons 2 ->
                   Vehicle Weapons 3 -> Vehicle Plating 3 -> Ship Weapons 3
    """
    armories = [u for u in obs.observation.raw_units
                if u.alliance == features.PlayerRelative.SELF
                and u.unit_type == units.Terran.Armory]
    if not armories:
        return actions.RAW_FUNCTIONS.no_op()
    
    armory = armories[0]
    state = _research_state['armory'] % 9
    
    if state == 0:
        _research_state['armory'] += 1
        return actions.RAW_FUNCTIONS.Research_TerranVehicleWeaponsLevel1_quick("now", armory.tag)
    elif state == 1:
        _research_state['armory'] += 1
        return actions.RAW_FUNCTIONS.Research_TerranVehicleAndShipPlatingLevel1_quick("now", armory.tag)
    elif state == 2:
        _research_state['armory'] += 1
        return actions.RAW_FUNCTIONS.Research_TerranShipWeaponsLevel1_quick("now", armory.tag)
    elif state == 3:
        _research_state['armory'] += 1
        return actions.RAW_FUNCTIONS.Research_TerranVehicleWeaponsLevel2_quick("now", armory.tag)
    elif state == 4:
        _research_state['armory'] += 1
        return actions.RAW_FUNCTIONS.Research_TerranVehicleAndShipPlatingLevel2_quick("now", armory.tag)
    elif state == 5:
        _research_state['armory'] += 1
        return actions.RAW_FUNCTIONS.Research_TerranShipWeaponsLevel2_quick("now", armory.tag)
    elif state == 6:
        _research_state['armory'] += 1
        return actions.RAW_FUNCTIONS.Research_TerranVehicleWeaponsLevel3_quick("now", armory.tag)
    elif state == 7:
        _research_state['armory'] += 1
        return actions.RAW_FUNCTIONS.Research_TerranVehicleAndShipPlatingLevel3_quick("now", armory.tag)
    else:
        _research_state['armory'] += 1
        return actions.RAW_FUNCTIONS.Research_TerranShipWeaponsLevel3_quick("now", armory.tag)


def research_ghost_academy(obs, _x=None, _y=None):
    """Research next available upgrade at Ghost Academy
    Cycles through: Personal Cloaking -> Enhanced Shockwaves
    """
    ghost_academies = [u for u in obs.observation.raw_units
                       if u.alliance == features.PlayerRelative.SELF
                       and u.unit_type == units.Terran.GhostAcademy]
    if not ghost_academies:
        return actions.RAW_FUNCTIONS.no_op()
    
    academy = ghost_academies[0]
    state = _research_state['ghost_academy'] % 2
    
    if state == 0:
        _research_state['ghost_academy'] += 1
        return actions.RAW_FUNCTIONS.Research_PersonalCloaking_quick("now", academy.tag)
    else:
        _research_state['ghost_academy'] += 1
        return actions.RAW_FUNCTIONS.Research_EnhancedShockwaves_quick("now", academy.tag)