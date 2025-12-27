"""
Base Building System - Tracks proper base construction with tiered buildings.
Forces AI to build complete, tightly-packed bases around mineral centroids.
"""

import numpy as np
from pysc2.lib import units, features


class BaseTracker:
    """
    Tracks building construction at each base location.
    Enforces proper base structure: anchor + production + accessory buildings.
    """
    
    # Building tier definitions
    TIER_1_ANCHOR = {units.Terran.CommandCenter, units.Terran.OrbitalCommand, 
                     units.Terran.PlanetaryFortress}
    
    TIER_2_PRODUCTION = {units.Terran.Barracks, units.Terran.Factory, 
                         units.Terran.Starport}
    
    TIER_3_ACCESSORY = {
        units.Terran.SupplyDepot, units.Terran.SupplyDepotLowered,
        units.Terran.Refinery, units.Terran.EngineeringBay,
        units.Terran.Armory, units.Terran.Bunker,
        units.Terran.MissileTurret, units.Terran.SensorTower,
        units.Terran.GhostAcademy, units.Terran.FusionCore
    }
    
    def __init__(self):
        # Store mineral centroids in expansion order
        self.expansion_centroids = []  # [(x, y), ...]
        self.base_completion = []  # [dict per base with completion status]
        
    def initialize_expansions(self, obs):
        """
        Find all mineral clusters and order them by distance from starting CC.
        Called once at episode start.
        """
        # Find starting CC location
        ccs = [u for u in obs.observation.raw_units
            if u.alliance == features.PlayerRelative.SELF 
            and u.unit_type in self.TIER_1_ANCHOR]
        
        if not ccs:
            return
        
        start_cc = ccs[0]
        start_pos = (start_cc.x, start_cc.y)
        
        # Find all mineral fields
        minerals = [u for u in obs.observation.raw_units
                if u.alliance == features.PlayerRelative.NEUTRAL
                and u.unit_type in (units.Neutral.MineralField, 
                                    units.Neutral.MineralField750,
                                    units.Neutral.LabMineralField,
                                    units.Neutral.LabMineralField750,
                                    units.Neutral.RichMineralField,
                                    units.Neutral.RichMineralField750)]
        
        # Cluster minerals into bases (group nearby minerals)
        clusters = self._cluster_minerals(minerals)
        
        # Sort clusters by distance from starting position
        clusters_with_dist = []
        for cluster in clusters:
            centroid = self._calculate_centroid(cluster)
            dist = np.sqrt((centroid[0] - start_pos[0])**2 + 
                        (centroid[1] - start_pos[1])**2)
            clusters_with_dist.append((dist, centroid))
        
        clusters_with_dist.sort(key=lambda x: x[0])
        
        # Store centroids in expansion order
        self.expansion_centroids = [c[1] for c in clusters_with_dist]
        
        # Initialize completion tracking for each base
        self.base_completion = []
        for i, _ in enumerate(self.expansion_centroids):
            # First base (index 0) starts with an anchor (the starting CC)
            self.base_completion.append({
                'anchor': 1 if i == 0 else 0,  # FIXED: Starting base has anchor
                'production': 0,
                'accessory': 0,
                'completed': False,
                'reward_given': False
            })
    
    def _cluster_minerals(self, minerals, cluster_distance=16):
        """Group minerals into base clusters - finds closest next cluster at any distance"""
        if not minerals:
            return []
        
        clusters = []
        remaining = list(minerals)
        
        while remaining:
            # Start new cluster with first remaining mineral
            seed = remaining.pop(0)
            cluster = [seed]
            
            # Add all minerals within cluster_distance of this seed
            i = 0
            while i < len(remaining):
                mineral = remaining[i]
                # Check distance to seed
                dist = np.sqrt((seed.x - mineral.x)**2 + (seed.y - mineral.y)**2)
                
                if dist < cluster_distance:
                    cluster.append(mineral)
                    remaining.pop(i)
                else:
                    i += 1
            
            clusters.append(cluster)
        
        return clusters
    
    def _calculate_centroid(self, mineral_cluster):
        """Calculate centroid of mineral cluster"""
        x = sum(m.x for m in mineral_cluster) / len(mineral_cluster)
        y = sum(m.y for m in mineral_cluster) / len(mineral_cluster)
        return (x, y)
    
    def update_base_status(self, obs):
        """
        Update building counts for each base.
        Counts buildings within radius of 8 from each centroid.
        """
        if not self.expansion_centroids:
            self.initialize_expansions(obs)
            return
        
        # Get all player buildings
        buildings = [u for u in obs.observation.raw_units
                    if u.alliance == features.PlayerRelative.SELF
                    and u.build_progress == 100]  # Only count completed buildings
        
        # Update each base
        for i, centroid in enumerate(self.expansion_centroids):
            anchor_count = 0
            production_count = 0
            accessory_count = 0
            
            # Count buildings within radius 8 of this centroid
            for building in buildings:
                dist = np.sqrt((building.x - centroid[0])**2 + 
                              (building.y - centroid[1])**2)
                
                if dist <= 12:
                    if building.unit_type in self.TIER_1_ANCHOR:
                        anchor_count += 1
                    elif building.unit_type in self.TIER_2_PRODUCTION:
                        production_count += 1
                    elif building.unit_type in self.TIER_3_ACCESSORY:
                        accessory_count += 1
            
            # Update counts
            self.base_completion[i]['anchor'] = anchor_count
            self.base_completion[i]['production'] = production_count
            self.base_completion[i]['accessory'] = accessory_count
            
            # Check completion: 1 anchor, 3 production, 5 accessory
            if (anchor_count >= 1 and 
                production_count >= 2 and 
                accessory_count >= 3):
                self.base_completion[i]['completed'] = True
    
    def can_expand(self, obs):
        """
        Check if we're allowed to build a new Command Center.
        Requirements: Current furthest base must be COMPLETE (1 anchor, 3 production, 5 accessory).
        """
        if not self.expansion_centroids:
            self.initialize_expansions(obs)
        
        # Find the furthest base that has an anchor (our current expansion)
        current_expansion_index = -1
        for i, status in enumerate(self.base_completion):
            if status['anchor'] >= 1:
                current_expansion_index = i
        
        # If we have no bases yet (shouldn't happen - starting CC should exist)
        if current_expansion_index < 0:
            return False
        
        # Check if current base is COMPLETE: 1 anchor, 3 production, 5 accessory
        current_base = self.base_completion[current_expansion_index]
        if (current_base['anchor'] < 1 or 
            current_base['production'] < 2 or  # Match completion
            current_base['accessory'] < 3):
            return False
        
        return True
    
    def get_available_expansions(self, obs):
        """
        Returns ALL expansion centroids that don't have anchors yet.
        Only returns expansions if can_expand() is True (current base is complete: 1/3/5).
        No limit on expansions - can build multiple CCs if prereqs met.
        """
        if not self.expansion_centroids:
            self.initialize_expansions(obs)
        
        # Check if we're allowed to expand
        if not self.can_expand(obs):
            return []
        
        # Return ALL bases without anchors
        available = []
        for i, status in enumerate(self.base_completion):
            if status['anchor'] == 0:
                available.append(self.expansion_centroids[i])
        
        return available



    def check_completion_rewards(self):
        """
        Check which bases completed this step and return total reward.
        Returns: (total_reward, [list of newly completed base indices])
        """
        total_reward = 0.0
        newly_completed = []
        
        # Tiered rewards: main=50, natural=40, third=30, fourth+=20
        reward_tiers = [50.0, 40.0, 30.0, 20.0]
        
        for i, status in enumerate(self.base_completion):
            if status['completed'] and not status['reward_given']:
                # Determine reward based on base number
                if i < len(reward_tiers):
                    reward = reward_tiers[i]
                else:
                    reward = 20.0  # Default for 4th+ bases
                
                total_reward += reward
                status['reward_given'] = True
                newly_completed.append(i)
        
        return total_reward, newly_completed
    
    def get_status_summary(self):
        """Get human-readable status of all bases"""
        summary = []
        base_names = ['Main', 'Natural', 'Third', 'Fourth', 'Fifth', 'Sixth']
        
        for i, status in enumerate(self.base_completion):
            name = base_names[i] if i < len(base_names) else f'Base {i+1}'
            summary.append(
                f"{name}: {status['anchor']}/1 anchor, "
                f"{status['production']}/3 production, "
                f"{status['accessory']}/5 accessory "
                f"{'✓ COMPLETE' if status['completed'] else ''}"
            )
        
        return '\n'.join(summary)
    
    def reset(self):
        """Reset for new episode"""
        self.expansion_centroids = []
        self.base_completion = []