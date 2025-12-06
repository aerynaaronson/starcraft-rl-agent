from pysc2.lib import actions, features, units
from pysc2.env import environment
from s2clientprotocol import raw_pb2
from s2clientprotocol import sc2api_pb2
from pysc2.lib.units import Terran

def do_nothing(obs, _x=None, _y=None):
    return actions.RAW_FUNCTIONS.no_op()
def send_idle_workers_to_mine(obs, _x=None, _y=None):
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
def choose_build_tile(obs, tile_x, tile_y):
    if 0 <= tile_x <= 64 and 0 <= tile_y <= 64:
        return tile_x, tile_y
    MINIMAP_W = 84
    MINIMAP_H = 84
    mx = tile_x / MINIMAP_W
    my = tile_y / MINIMAP_H
    MAP_W = 64
    MAP_H = 64
    world_x = mx * MAP_W
    world_y = my * MAP_H
    return world_x, world_y
def train_worker(obs, _x=None, _y=None):
    townhalls = [
        u for u in obs.observation.raw_units
        if u.alliance == features.PlayerRelative.SELF and
           u.unit_type == units.Terran.CommandCenter
    ]
    if not townhalls:
        return actions.RAW_FUNCTIONS.no_op()
    cc = townhalls[0]
    return actions.RAW_FUNCTIONS.Train_SCV_quick("now", cc.tag)
def build_refinery(obs):
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
    closest = min(geysers, key=lambda g: (g.x - scv.x)**2 + (g.y - scv.y)**2)
    return actions.RAW_FUNCTIONS.Build_Refinery_pt(
        "now",
        [scv.tag],          
        [closest.tag]      
    )
def build_refinery_fn(obs, x=None, y=None):
    return build_refinery(obs)
def send_idle_workers_to_gas(obs, _x=None, _y=None):
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
def train_marine(obs, _x=None, _y=None):
    barracks = [
        u for u in obs.observation.raw_units
        if u.alliance == features.PlayerRelative.SELF and
           u.unit_type == units.Terran.Barracks
    ]
    if not barracks:
        return actions.RAW_FUNCTIONS.no_op()
    rax = barracks[0]
    return actions.RAW_FUNCTIONS.Train_Marine_quick("now", rax.tag)
def build_supply_depot(obs, tile_x, tile_y):
    scvs = [u for u in obs.observation.raw_units 
            if u.unit_type == units.Terran.SCV and u.alliance == features.PlayerRelative.SELF]
    if not scvs:
        return actions.RAW_FUNCTIONS.no_op()
    scv = scvs[0]
    wx, wy = choose_build_tile(obs, tile_x, tile_y)
    return actions.RAW_FUNCTIONS.Build_SupplyDepot_pt(
        "now",
        scv.tag,
        (wx, wy)
    )
def build_barracks(obs, tile_x, tile_y):
    wx, wy = choose_build_tile(obs, tile_x, tile_y)
    scvs = [
        u for u in obs.observation.raw_units
        if u.alliance == features.PlayerRelative.SELF and
           u.unit_type == units.Terran.SCV
    ]
    if not scvs:
        return actions.RAW_FUNCTIONS.no_op()
    scv = scvs[0]
    return actions.RAW_FUNCTIONS.Build_Barracks_pt(
        "now",
        scv.tag,
        (wx, wy)
    )
def build_factory(obs, tile_x, tile_y):
    wx, wy = choose_build_tile(obs, tile_x, tile_y)
    scvs = [
        u for u in obs.observation.raw_units
        if u.alliance == features.PlayerRelative.SELF and
           u.unit_type == units.Terran.SCV
    ]
    if not scvs:
        return actions.RAW_FUNCTIONS.no_op()
    scv = scvs[0]
    return actions.RAW_FUNCTIONS.Build_Factory_pt(
        "now",
        scv.tag,
        (wx, wy)
    )

def add_barracks_techlab(obs, _x=None, _y=None):
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

def train_hellion(obs, _x=None, _y=None):
    factories = [u for u in obs.observation.raw_units
                 if u.alliance == features.PlayerRelative.SELF
                 and u.unit_type == units.Terran.Factory]
    if not factories:
        return actions.RAW_FUNCTIONS.no_op()
    factory = factories[0]
    return actions.RAW_FUNCTIONS.Train_Hellion_quick("now", factory.tag)

def train_marauder(obs, _x=None, _y=None):
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
    barracks = [
        u for u in obs.observation.raw_units
        if u.alliance == features.PlayerRelative.SELF and
           u.unit_type == units.Terran.Barracks
    ]
    if not barracks:
        return actions.RAW_FUNCTIONS.no_op()
    rax = barracks[0]
    return actions.RAW_FUNCTIONS.Train_Reaper_quick("now", rax.tag)

def build_armory(obs, tile_x=None, tile_y=None):
    scvs = [u for u in obs.observation.raw_units
            if u.alliance == features.PlayerRelative.SELF and u.unit_type == units.Terran.SCV]
    if not scvs:
        return actions.RAW_FUNCTIONS.no_op()
    scv = scvs[0]
    if tile_x is None or tile_y is None:
        tile_x, tile_y = 20, 20
    wx, wy = choose_build_tile(obs, tile_x, tile_y)
    return actions.RAW_FUNCTIONS.Build_Armory_pt("now", scv.tag, (wx, wy))

def build_starport(obs, tile_x, tile_y):
    wx, wy = choose_build_tile(obs, tile_x, tile_y)
    scvs = [
        u for u in obs.observation.raw_units
        if u.alliance == features.PlayerRelative.SELF and
           u.unit_type == units.Terran.SCV
    ]
    if not scvs:
        return actions.RAW_FUNCTIONS.no_op()
    scv = scvs[0]
    return actions.RAW_FUNCTIONS.Build_Starport_pt(
        "now",
        scv.tag,
        (wx, wy)
    )

def build_fusion_core(obs, tile_x=None, tile_y=None):
    scvs = [u for u in obs.observation.raw_units
            if u.alliance == features.PlayerRelative.SELF and u.unit_type == units.Terran.SCV]
    if not scvs:
        return actions.RAW_FUNCTIONS.no_op()
    scv = scvs[0]
    if tile_x is None or tile_y is None:
        tile_x, tile_y = 20, 20
    wx, wy = choose_build_tile(obs, tile_x, tile_y)
    return actions.RAW_FUNCTIONS.Build_FusionCore_pt("now", scv.tag, (wx, wy))

def add_starport_techlab(obs, _x=None, _y=None):
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

def train_siege_tank(obs, _x=None, _y=None):
    factories = [u for u in obs.observation.raw_units
                 if u.alliance == features.PlayerRelative.SELF
                 and u.unit_type == units.Terran.Factory]
    if not factories:
        return actions.RAW_FUNCTIONS.no_op()
    factory = factories[0]
    return actions.RAW_FUNCTIONS.Train_SiegeTank_quick("now", factory.tag)

def train_thor(obs, _x=None, _y=None):
    factories = [u for u in obs.observation.raw_units
                 if u.alliance == features.PlayerRelative.SELF
                 and u.unit_type == units.Terran.Factory]
    if not factories:
        return actions.RAW_FUNCTIONS.no_op()
    factory = factories[0]
    return actions.RAW_FUNCTIONS.Train_Thor_quick("now", factory.tag)

def train_cyclone(obs, _x=None, _y=None):
    factories = [u for u in obs.observation.raw_units
                 if u.alliance == features.PlayerRelative.SELF
                 and u.unit_type == units.Terran.Factory]
    if not factories:
        return actions.RAW_FUNCTIONS.no_op()
    factory = factories[0]
    return actions.RAW_FUNCTIONS.Train_Cyclone_quick("now", factory.tag)

def train_viking(obs, _x=None, _y=None):
    starports = [u for u in obs.observation.raw_units
                 if u.alliance == features.PlayerRelative.SELF
                 and u.unit_type == units.Terran.Starport]
    if not starports:
        return actions.RAW_FUNCTIONS.no_op()
    starport = starports[0]
    return actions.RAW_FUNCTIONS.Train_VikingFighter_quick("now", starport.tag)

def train_medivac(obs, _x=None, _y=None):
    starports = [u for u in obs.observation.raw_units
                 if u.alliance == features.PlayerRelative.SELF
                 and u.unit_type == units.Terran.Starport]
    if not starports:
        return actions.RAW_FUNCTIONS.no_op()
    starport = starports[0]
    return actions.RAW_FUNCTIONS.Train_Medivac_quick("now", starport.tag)

def train_liberator(obs, _x=None, _y=None):
    starports = [u for u in obs.observation.raw_units
                 if u.alliance == features.PlayerRelative.SELF
                 and u.unit_type == units.Terran.Starport]
    if not starports:
        return actions.RAW_FUNCTIONS.no_op()
    starport = starports[0]
    return actions.RAW_FUNCTIONS.Train_Liberator_quick("now", starport.tag)

def train_raven(obs, _x=None, _y=None):
    starports = [u for u in obs.observation.raw_units
                 if u.alliance == features.PlayerRelative.SELF
                 and u.unit_type == units.Terran.Starport]
    if not starports:
        return actions.RAW_FUNCTIONS.no_op()
    starport = starports[0]
    return actions.RAW_FUNCTIONS.Train_Raven_quick("now", starport.tag)

def train_banshee(obs, _x=None, _y=None):
    starports = [u for u in obs.observation.raw_units
                 if u.alliance == features.PlayerRelative.SELF
                 and u.unit_type == units.Terran.Starport]
    if not starports:
        return actions.RAW_FUNCTIONS.no_op()
    starport = starports[0]
    return actions.RAW_FUNCTIONS.Train_Banshee_quick("now", starport.tag)

def train_battlecruiser(obs, _x=None, _y=None):
    starports = [u for u in obs.observation.raw_units
                 if u.alliance == features.PlayerRelative.SELF
                 and u.unit_type == units.Terran.Starport]
    if not starports:
        return actions.RAW_FUNCTIONS.no_op()
    starport = starports[0]
    return actions.RAW_FUNCTIONS.Train_Battlecruiser_quick("now", starport.tag)

def build_command_center(obs, tile_x, tile_y):
    wx, wy = choose_build_tile(obs, tile_x, tile_y)
    scvs = [
        u for u in obs.observation.raw_units
        if u.alliance == features.PlayerRelative.SELF and
           u.unit_type == units.Terran.SCV
    ]
    if not scvs:
        return actions.RAW_FUNCTIONS.no_op()
    scv = scvs[0]
    return actions.RAW_FUNCTIONS.Build_CommandCenter_pt(
        "now",
        scv.tag,
        (wx, wy)
    )
def lower_depot(obs, _x=None, _y=None):
    depots = [
        u for u in obs.observation.raw_units
        if u.alliance == features.PlayerRelative.SELF and
           u.unit_type == units.Terran.SupplyDepot
    ]
    if not depots:
        return actions.RAW_FUNCTIONS.no_op()
    depot = depots[0]
    return actions.RAW_FUNCTIONS.Morph_SupplyDepot_Lower_quick("now", depot.tag)
#def liftoff(obs, _x=None, _y=None):
 #   return actions.RAW_FUNCTIONS.no_op()
#def land(obs, _x=None, _y=None):
 #   return actions.RAW_FUNCTIONS.no_op()
def attack_move(obs, target_x, target_y):
    army_units = [
        u for u in obs.observation.raw_units
        if u.alliance == features.PlayerRelative.SELF and
           u.unit_type not in (units.Terran.SCV, units.Terran.MULE)
    ]
    if not army_units:
        return actions.RAW_FUNCTIONS.no_op()
    enemy_units = [
        u for u in obs.observation.raw_units
        if u.alliance == features.PlayerRelative.ENEMY
    ]
    if enemy_units:
        commands = []
        for unit in army_units:
            closest_enemy = min(enemy_units,
                                key=lambda e: (e.x - unit.x)**2 + (e.y - unit.y)**2)
            commands.append(
                actions.RAW_FUNCTIONS.Attack_unit("now", unit.tag, closest_enemy.tag)
            )
        return commands
    if 0 <= target_x <= 1 and 0 <= target_y <= 1:
        map_w, map_h = 63, 63 
        world_x = int(target_x * map_w)
        world_y = int(target_y * map_h)
    else:
        world_x, world_y = int(target_x), int(target_y)
    return [actions.RAW_FUNCTIONS.Attack_pt("now", unit.tag, (world_x, world_y))
            for unit in army_units]
#def defend(obs, x, y):
 #   return actions.RAW_FUNCTIONS.no_op()
#def retreat(obs, x, y):
 #   return actions.RAW_FUNCTIONS.no_op()
#def scout_with_scv(obs, target_x, target_y):
 #   return actions.RAW_FUNCTIONS.no_op()
#def salvage_bunker(obs, _x=None, _y=None):
 #   return actions.RAW_FUNCTIONS.no_op()
ACTION_NEEDS_COORDS = {
    "do_nothing": False,
    "send_idle_workers_to_mine": False,
    "train_worker": False,
    "train_marine": False,
    "build_supply_depot": True,
    "build_barracks": True,
    "build_command_center": True,
    "lower_depot": False,
#    "raise_depot": False,
#    "liftoff": False,
#    "land": True,
    "attack_move": True,
#    "defend": True,
#    "retreat": True,
#    "scout_with_scv": True,
#    "salvage_bunker": False,
    "build_refinery": True,
    "send_idle_workers_to_gas": False,
    "add_barracks_techlab": False,
    "add_barracks_reactor": False,
    "build_techlab": False,
    "build_reactor": False,
    "train_marauder": False,
    "train_reaper": False,
    "build_factory": True,
    "train_hellion": False,
    "add_factory_techlab": False,
    "add_factory_reactor": False,
    "build_armory": True,
    "train_siege_tank": False,
    "train_thor": False,
    "train_cyclone": False,
    "build_starport": True,
    "build_fusion_core": True,
    "add_starport_techlab": False,
    "add_starport_reactor": False,
    "train_viking": False,
    "train_medivac": False,
    "train_liberator": False,
    "train_raven": False,
    "train_banshee": False,
    "train_battlecruiser": False,
}
ACTION_REGISTRY = {
    "do_nothing": do_nothing,
    "send_idle_workers_to_mine": send_idle_workers_to_mine,
    "train_worker": train_worker,
    "train_marine": train_marine,
    "build_supply_depot": build_supply_depot,
    "build_barracks": build_barracks,
    "build_command_center": build_command_center,
    "lower_depot": lower_depot,
#    "raise_depot": raise_depot,
#    "liftoff": liftoff,
#    "land": land,
    "attack_move": attack_move,
#    "defend": defend,
#    "retreat": retreat,
#    "scout_with_scv": scout_with_scv,
#    "salvage_bunker": salvage_bunker,
    "build_refinery": build_refinery_fn,
    "send_idle_workers_to_gas": send_idle_workers_to_gas,
    "add_barracks_techlab": add_barracks_techlab,
    "add_barracks_reactor": add_barracks_reactor,
    "build_techlab": add_factory_techlab,
    "build_reactor": add_factory_reactor,
    "train_marauder": train_marauder,
    "train_reaper": train_reaper,
    "build_factory": build_factory,
    "train_hellion": train_hellion,
    "add_factory_techlab": add_factory_techlab,
    "add_factory_reactor": add_factory_reactor,
    "build_armory": build_armory,         
    "train_siege_tank": train_siege_tank,  
    "train_thor": train_thor,              
    "train_cyclone": train_cyclone,
    "build_starport": build_starport,
    "build_fusion_core": build_fusion_core,
    "add_starport_techlab": add_starport_techlab,
    "add_starport_reactor": add_starport_reactor,
    "train_viking": train_viking,
    "train_medivac": train_medivac,
    "train_liberator": train_liberator,
    "train_raven": train_raven,
    "train_banshee": train_banshee,
    "train_battlecruiser": train_battlecruiser,
}