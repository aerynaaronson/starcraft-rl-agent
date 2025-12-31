import numpy as np
from collections import deque
from base_building_system import BaseTracker


class TieredRewardCalculator:
    """
    Instantaneous reward system - rewards given ONCE at the moment of achievement.

    Tier 1: Economy & Base Building (0 to 0.99 max)
    Tier 2: Army Production & Tech (0 to 0.99 max)
    Tier 3: Combat Efficiency (0 to 0.99 max) - FINAL TIER

    No terminal bonuses - all rewards are milestone-based only.
    No negative rewards - only positive reinforcement for achievements.
    """
    
    def __init__(self, worker_id=0, base_tracker=None):
        self.worker_id = worker_id
        self.current_tier = 1
        self.episode_rewards_history = deque(maxlen=100)
        self.episodes_in_tier = 0
        self.min_episodes_per_tier = 25
        self.base_tracker = base_tracker if base_tracker is not None else BaseTracker()
        
        self.tier_scores = []
        self.graduation_threshold = 0.85
        self.graduation_window = 10
        
        # Track what's been rewarded THIS episode (reset each episode)
        self.awarded = set()
        
        # Running total of instantaneous rewards this episode
        self.episode_instant_total = 0.0
        
        # Previous state for delta detection
        self.prev_state = {}
        
        # Enemy tracking for T3
        self.prev_enemy_buildings = 0
        self.prev_enemy_army = 0
    
    def reset(self):
        """Reset for new episode."""
        self.awarded = set()
        self.episode_instant_total = 0.0
        self.prev_state = {}
        self.prev_enemy_buildings = 0
        self.prev_enemy_army = 0
        self.base_tracker.reset()
        self._total_buildings_destroyed = 0
        # T3 combat tracking
        self._prev_our_army = 0
        self._prev_our_workers = 0
        self._total_damage_dealt = 0.0
        self._total_damage_taken = 0.0
    
    def calculate_step_reward(self, perception, action_name, obs, world_x=None, world_y=None):
        """
        Calculate instantaneous reward for this step.
        Returns non-zero ONLY when a milestone is achieved for the first time.

        Args:
            perception: GamePerception object
            action_name: Name of action taken
            obs: Raw observation
            world_x, world_y: Unused (kept for backwards compatibility)
        """
        self.base_tracker.update_base_status(obs)
        current = self._extract_state(perception)
        
        # Initialize prev_state on first call
        if not self.prev_state:
            self.prev_state = current.copy()
            self.prev_enemy_buildings = self._get_enemy_buildings(perception)
            self.prev_enemy_army = self._get_enemy_army(perception)
            return 0.0
        
        reward = 0.0

        if self.current_tier == 1:
            reward = self._tier1_instant(current, action_name, obs)
        elif self.current_tier == 2:
            reward = self._tier2_instant(current, action_name, perception)
        elif self.current_tier == 3:
            reward = self._tier3_instant(current, action_name, perception)

        # Building placement bonuses removed - milestone rewards only

        self.prev_state = current.copy()
        self.episode_instant_total += reward

        return reward
    
    def _tier1_instant(self, current, action_name, obs):
        """
        Tier 1: Economy & Base Building
        Rewards for: workers, supply depots, production buildings, expansions, addons, tech
        Scaled so maximum possible = 0.99 (no terminal bonus)
        """
        reward = 0.0

        # === WORKER MILESTONES ===
        scv = current['scv_count']
        for threshold, r in [(16, 0.018), (22, 0.027), (30, 0.027), (44, 0.036), (55, 0.036), (66, 0.045)]:
            key = f"workers_{threshold}"
            if scv >= threshold and key not in self.awarded:
                reward += r
                self.awarded.add(key)

        # === SUPPLY DEPOTS (first 4) ===
        if action_name == "build_supply_depot":
            depot_count = len([k for k in self.awarded if k.startswith("depot_")])
            if depot_count < 4:
                reward += 0.018
                self.awarded.add(f"depot_{depot_count}")

        # === PRODUCTION BUILDINGS ===
        # First barracks
        if current['barracks_count'] >= 1 and "first_barracks" not in self.awarded:
            reward += 0.045
            self.awarded.add("first_barracks")

        # Second barracks
        if current['barracks_count'] >= 2 and "second_barracks" not in self.awarded:
            reward += 0.027
            self.awarded.add("second_barracks")

        # Third+ barracks
        if current['barracks_count'] >= 3 and "third_barracks" not in self.awarded:
            reward += 0.018
            self.awarded.add("third_barracks")

        # First factory
        if current['factory_count'] >= 1 and "first_factory" not in self.awarded:
            reward += 0.036
            self.awarded.add("first_factory")

        # First starport
        if current['starport_count'] >= 1 and "first_starport" not in self.awarded:
            reward += 0.036
            self.awarded.add("first_starport")

        # === EXPANSION ===
        if current['cc_count'] >= 2 and "expansion" not in self.awarded:
            reward += 0.073
            self.awarded.add("expansion")

        # Third base
        if current['cc_count'] >= 3 and "third_base" not in self.awarded:
            reward += 0.054
            self.awarded.add("third_base")

        # === BASE COMPLETION (from BaseTracker) ===
        bases_completed = sum(1 for b in self.base_tracker.base_completion if b.get('completed', False))
        if bases_completed >= 1 and "base_complete_1" not in self.awarded:
            reward += 0.091
            self.awarded.add("base_complete_1")
        if bases_completed >= 2 and "base_complete_2" not in self.awarded:
            reward += 0.109
            self.awarded.add("base_complete_2")

        # === ADDONS ===
        if current['techlab_count'] >= 1 and "first_techlab" not in self.awarded:
            reward += 0.027
            self.awarded.add("first_techlab")
        if current['reactor_count'] >= 1 and "first_reactor" not in self.awarded:
            reward += 0.027
            self.awarded.add("first_reactor")
        if current['techlab_count'] + current['reactor_count'] >= 3 and "three_addons" not in self.awarded:
            reward += 0.027
            self.awarded.add("three_addons")

        # === TECH BUILDINGS ===
        if current['engineering_bay_count'] >= 1 and "ebay" not in self.awarded:
            reward += 0.027
            self.awarded.add("ebay")
        if current['armory_count'] >= 1 and "armory" not in self.awarded:
            reward += 0.018
            self.awarded.add("armory")

        # === ORBITAL COMMAND ===
        if action_name == "upgrade_orbital_command" and "first_orbital" not in self.awarded:
            reward += 0.036
            self.awarded.add("first_orbital")

        # === REFINERY ===
        if action_name == "build_refinery":
            ref_count = len([k for k in self.awarded if k.startswith("refinery_")])
            if ref_count < 4:
                reward += 0.018
                self.awarded.add(f"refinery_{ref_count}")

        return reward
    
    def _tier2_instant(self, current, action_name, perception):
        """
        Tier 2: Army Production & Tech
        Rewards for: army supply milestones, unit diversity, upgrades, research
        Scaled so maximum possible = 0.99 (no terminal bonus)

        Distribution: T1 Economy 30%, Army Size 30%, Diversity 20%, Upgrades 10%, Research 10%
        """
        reward = 0.0

        # === T1 ECONOMY MAINTENANCE (30% of total) ===
        scv = current['scv_count']
        for threshold, r in [(44, 0.099), (55, 0.099), (66, 0.099)]:
            key = f"t2_workers_{threshold}"
            if scv >= threshold and key not in self.awarded:
                reward += r
                self.awarded.add(key)

        # === ARMY SUPPLY MILESTONES (30% of total) ===
        army = current['army_supply']
        for threshold, r in [(10, 0.030), (20, 0.040), (30, 0.040), (50, 0.050), (80, 0.055), (100, 0.045), (130, 0.037)]:
            key = f"army_{threshold}"
            if army >= threshold and key not in self.awarded:
                reward += r
                self.awarded.add(key)

        # === UNIT DIVERSITY (20% of total) ===
        unit_types_made = sum(1 for u, c in current['unit_counts'].items() if c > 0)
        for threshold, r in [(2, 0.040), (4, 0.055), (6, 0.055), (8, 0.050)]:
            key = f"unit_types_{threshold}"
            if unit_types_made >= threshold and key not in self.awarded:
                reward += r
                self.awarded.add(key)

        # === UPGRADES (10% of total) ===
        upgrades = current['upgrades']
        for threshold, r in [(1, 0.020), (2, 0.020), (3, 0.020), (4, 0.020), (6, 0.019)]:
            key = f"upgrades_{threshold}"
            if upgrades >= threshold and key not in self.awarded:
                reward += r
                self.awarded.add(key)

        # === RESEARCH ACTIONS (10% of total - capped at 5 actions) ===
        research_count = len([k for k in self.awarded if k.startswith("research_")])
        if action_name.startswith("research_") and action_name not in self.awarded and research_count < 5:
            reward += 0.020
            self.awarded.add(action_name)

        return reward
    
    def _tier3_instant(self, current, action_name, perception):
        """
        Tier 3: Combat Efficiency (FINAL TIER)

        Tracks combat milestones and damage dealt.
        No terminal bonus - purely combat achievement based.
        Scaled so maximum possible = 0.99 (no negative rewards)
        """
        reward = 0.0

        current_enemy_buildings = self._get_enemy_buildings(perception)
        current_enemy_army = self._get_enemy_army(perception)

        # Track our losses too (for potential future use)
        our_army = current['army_supply']
        our_workers = current['scv_count']

        if not hasattr(self, '_prev_our_army'):
            self._prev_our_army = our_army
            self._prev_our_workers = our_workers
            self._total_damage_dealt = 0.0
            self._total_damage_taken = 0.0

        # === DAMAGE DEALT (positive) ===
        # Enemy buildings destroyed (scaled down from 0.05 to 0.045)
        if current_enemy_buildings < self.prev_enemy_buildings:
            destroyed = self.prev_enemy_buildings - current_enemy_buildings
            damage = destroyed * 0.045
            reward += damage
            self._total_damage_dealt += damage

        # Enemy army killed (scaled down from 0.03 to 0.027)
        if current_enemy_army < self.prev_enemy_army:
            killed = self.prev_enemy_army - current_enemy_army
            damage = (killed / 10.0) * 0.027  # Per ~10 supply
            reward += damage
            self._total_damage_dealt += damage

        # === DAMAGE TAKEN (track but don't penalize - no negative rewards) ===
        # We lost army
        if our_army < self._prev_our_army:
            lost_army = self._prev_our_army - our_army
            self._total_damage_taken += (lost_army / 10.0) * 0.027

        # We lost workers
        if our_workers < self._prev_our_workers:
            lost_workers = self._prev_our_workers - our_workers
            self._total_damage_taken += lost_workers * 0.009

        # === DESTRUCTION MILESTONES ===
        total_destroyed = getattr(self, '_total_buildings_destroyed', 0)
        if current_enemy_buildings < self.prev_enemy_buildings:
            total_destroyed += (self.prev_enemy_buildings - current_enemy_buildings)
            self._total_buildings_destroyed = total_destroyed

        for threshold, r in [(3, 0.045), (6, 0.054), (10, 0.073)]:
            key = f"destroyed_{threshold}"
            if total_destroyed >= threshold and key not in self.awarded:
                reward += r
                self.awarded.add(key)

        # === ATTACK ACTIONS ===
        attack_actions = ["attack_aggressor", "attack_siege", "attack_support", "attack_harass", "final_push"]
        if action_name in attack_actions:
            attack_count = len([k for k in self.awarded if k.startswith("attack_")])
            if attack_count < 5:
                reward += 0.018
                self.awarded.add(f"attack_{attack_count}")

        # Update tracking
        self.prev_enemy_buildings = current_enemy_buildings
        self.prev_enemy_army = current_enemy_army
        self._prev_our_army = our_army
        self._prev_our_workers = our_workers

        return reward
    
    
    def _extract_state(self, p):
        """Extract current game state from perception."""
        unit_counts = {
            'marine': p.marine_count(),
            'marauder': p.marauder_count(),
            'reaper': p.reaper_count(),
            'ghost': p.ghost_count(),
            'hellion': p.hellion_count(),
            'hellbat': p.hellbat_count(),
            'siege_tank': p.siege_tank_count(),
            'thor': p.thor_count(),
            'cyclone': p.cyclone_count(),
            'widow_mine': p.widow_mine_count(),
            'viking': p.viking_count(),
            'medivac': p.medivac_count(),
            'liberator': p.liberator_count(),
            'raven': p.raven_count(),
            'banshee': p.banshee_count(),
            'battlecruiser': p.battlecruiser_count(),
        }
        
        upgrades = sum([
            1 if p.has_stimpack() else 0,
            1 if p.has_combat_shield() else 0,
            1 if p.has_concussive_shells() else 0,
            1 if p.has_siege_tech() else 0,
            1 if p.has_infantry_weapons_1() else 0,
            1 if p.has_infantry_armor_1() else 0,
        ])
        
        return {
            'scv_count': p.scv_count(),
            'cc_count': p.cc_count(),
            'barracks_count': p.barracks_count(),
            'factory_count': p.factory_count(),
            'starport_count': p.starport_count(),
            'engineering_bay_count': p.engineering_bay_count(),
            'armory_count': p.armory_count(),
            'fusion_core_count': p.fusion_core_count(),
            'ghost_academy_count': p.ghost_academy_count(),
            'techlab_count': p.barracks_techlab_count() + p.factory_techlab_count() + p.starport_techlab_count(),
            'reactor_count': p.barracks_reactor_count() + p.factory_reactor_count() + p.starport_reactor_count(),
            'army_supply': p.army_supply(),
            'total_supply': p.supply_used if hasattr(p, 'supply_used') else p.army_supply() + p.scv_count(),
            'unit_counts': unit_counts,
            'upgrades': upgrades,
        }
    
    def _get_enemy_buildings(self, p):
        if hasattr(p, 'enemy_buildings'):
            buildings = p.enemy_buildings
            return len(buildings) if hasattr(buildings, '__len__') else buildings()
        return 0
    
    def _get_enemy_army(self, p):
        if hasattr(p, 'enemy_army_supply'):
            return p.enemy_army_supply()
        return 0
    
    def calculate_terminal_reward(self, outcome, game_length):
        """
        Terminal reward - REMOVED for all tiers.
        All rewards are milestone-based only.
        """
        # No terminal bonuses - all tiers reward milestones only
        return 0.0
    
    def get_episode_score(self):
        """
        Get the total score for this episode.
        This is what determines tier graduation.
        All tiers: cap at 1.0 for safety (should be 0.99 max from milestones)
        """
        return min(self.episode_instant_total, 1.0)
    
    def _check_graduation(self):
        if len(self.episode_rewards_history) < 20:
            return False
        recent = list(self.episode_rewards_history)[-20:]
        avg = np.mean(recent)
        return avg >= self.graduation_threshold
    
    def end_episode(self, outcome, game_length=0):
        """End episode and check for tier graduation."""
        # No terminal bonuses - episode score is just the milestone rewards
        episode_score = self.get_episode_score()

        self.episode_rewards_history.append(episode_score)
        self.tier_scores.append(episode_score)
        self.episodes_in_tier += 1

        if len(self.tier_scores) > self.graduation_window:
            self.tier_scores = self.tier_scores[-self.graduation_window:]

        graduated = False

        # Only graduate from T1 and T2 (T3 is final)
        if self.current_tier < 3 and self.episodes_in_tier >= self.min_episodes_per_tier:
            if self._check_graduation():
                self.current_tier += 1
                self.episode_rewards_history.clear()
                self.tier_scores = []
                self.episodes_in_tier = 0
                graduated = True
                print(f"[Worker {self.worker_id}] GRADUATED to Tier {self.current_tier}")

        return graduated, self.current_tier, episode_score
    
    def get_status(self):
        avg_reward = np.mean(list(self.episode_rewards_history)) if self.episode_rewards_history else 0.0
        return f"T{self.current_tier} Ep{self.episodes_in_tier}/{self.min_episodes_per_tier} AvgR:{avg_reward:.2f}"


def compute_discounted_rewards(step_rewards, final_reward, gamma=0.997):
    """
    For instantaneous rewards, we don't need heavy discounting.
    Each step gets its own instant reward.
    No terminal bonuses - final_reward is always 0.0.
    """
    n = len(step_rewards)
    if n == 0:
        return [0.0]

    # Step rewards are already instantaneous - just return them as-is
    # No terminal bonus distribution since final_reward is always 0
    return list(step_rewards)


# Backwards compatibility
class RewardCalculator(TieredRewardCalculator):
    pass