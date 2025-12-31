from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from game_perception import GamePerception, encode_state
from terran_actions import ACTION_REGISTRY, ACTION_NEEDS_COORDS
from terran_legal_actions_logic import get_legal_actions
from lstm_memory import ACTION_TO_INDEX
from metrics_logger import MetricsLogger
from coordinate_transform import CoordinateTransform
from worker_rewards import TieredRewardCalculator, compute_discounted_rewards
from base_building_system import BaseTracker
import numpy as np
import random
import time as time_module
import queue


def create_env(worker_id, map_name):
    """Create SC2 environment"""
    from absl import flags
    FLAGS = flags.FLAGS
    if not FLAGS.is_parsed():
        FLAGS(['worker'])
    
    return sc2_env.SC2Env(
        map_name=map_name,
        players=[
            sc2_env.Agent(sc2_env.Race.terran),
            sc2_env.Bot(sc2_env.Race.random, sc2_env.Difficulty.very_easy)
        ],
        agent_interface_format=sc2_env.AgentInterfaceFormat(
            feature_dimensions=sc2_env.Dimensions(screen=84, minimap=64),
            use_feature_units=True,
            use_raw_units=True,
            action_space=actions.ActionSpace.RAW
        ),
        step_mul=8,
        game_steps_per_episode=0,
        visualize=False,
        random_seed=worker_id,
    )


def game_worker(worker_id, prediction_queue, result_queue, update_queue, episodes_per_worker, map_name='Simple64'):
    """
    Game worker with tiered curriculum training.
    
    Each worker tracks its own tier and graduates independently.
    FIXED: Now stores legal_actions for proper masking during training.
    """
    max_steps = 10_000
    epsilon_start = 0.3
    epsilon_end = 0.05
    epsilon_decay = 0.999
    
    GAMMA = 0.997
    
    print(f"Worker {worker_id} starting (Tiered Curriculum)...")
    print(f"  Map: {map_name}, Max steps: {max_steps}")
    
    logger = MetricsLogger(f'training_metrics_worker_{worker_id}.jsonl')
    env = create_env(worker_id, map_name)
    
    epsilon = epsilon_start
    base_tracker = BaseTracker()
    
    # Create tiered reward calculator for this worker
    reward_calc = TieredRewardCalculator(worker_id=worker_id)
    
    for episode in range(episodes_per_worker):
        base_tracker.reset()
        reward_calc.reset()
        obs = env.reset()[0]
        
        # Reset episode memory on GPU server
        prediction_queue.put({
            'command': 'RESET_EPISODE',
            'worker_id': worker_id
        })
        time_module.sleep(0.01)
        
        # Initialize episode
        perception = GamePerception(obs)
        cc_x, cc_y = perception.get_base_location()
        map_height, map_width = perception.get_map_size()
        coord_transform = CoordinateTransform(map_width, map_height, cc_x, cc_y)
        
        if episode == 0:
            info = coord_transform.get_map_info()
            print(f"Worker {worker_id} coordinate system:")
            print(f"  Map size: {info['map_size']}")
            print(f"  CC world position: {info['cc_world_pos']}")
            print(f"  Needs rotation: {info['needs_rotation']}")
        
        logger.start_episode(episode + 1, perception.get_start_corner())
        
        episode_experiences = []
        step_rewards = []
        done = False
        step_count = 0
        
        while not done and step_count < max_steps:
            if not base_tracker.expansion_centroids:
                base_tracker.initialize_expansions(obs)
            
            base_tracker.update_base_status(obs)
            
            perception.obs = obs
            perception.update_enemy_info(obs)
            
            state = encode_state(perception, coord_transform)
            
            legal_actions = get_legal_actions(obs)
            legal_actions = [a for a in legal_actions if a in ACTION_REGISTRY]
            if not legal_actions:
                legal_actions = ["do_nothing"]
            
            # Convert legal actions to indices for training
            legal_action_indices = [ACTION_TO_INDEX[a] for a in legal_actions if a in ACTION_TO_INDEX]
            if not legal_action_indices:
                legal_action_indices = [ACTION_TO_INDEX.get("do_nothing", 0)]
            
            log_prob = 0.0
            
            if random.random() < epsilon:
                # Epsilon exploration with tier-aware weighting
                if reward_calc.current_tier == 1:
                    # Tier 1: favor economy and base building
                    action_weights = {
                        # Workers & economy
                        "train_worker": 0.15,
                        "send_idle_workers_to_mine": 0.08,
                        "send_idle_workers_to_gas": 0.04,
                        "call_down_mule": 0.05,
                        
                        # Infrastructure
                        "build_supply_depot": 0.12,
                        "build_refinery": 0.06,
                        "build_command_center": 0.08,
                        
                        # Production buildings
                        "build_barracks": 0.10,
                        "build_factory": 0.06,
                        "build_starport": 0.04,
                        
                        # Tech buildings
                        "build_engineering_bay": 0.03,
                        "build_armory": 0.02,
                        "build_fusion_core": 0.01,
                        "build_ghost_academy": 0.01,
                        
                        # Defense buildings
                        "build_bunker": 0.02,
                        "build_missile_turret": 0.01,
                        "build_sensor_tower": 0.01,
                        
                        # Addons
                        "add_barracks_techlab": 0.02,
                        "add_barracks_reactor": 0.02,
                        "add_factory_techlab": 0.01,
                        "add_factory_reactor": 0.01,
                        "add_starport_techlab": 0.01,
                        "add_starport_reactor": 0.01,
                        
                        # CC upgrades
                        "upgrade_orbital_command": 0.03,
                        "lower_depot": 0.02,
                        
                        "do_nothing": 0.01,
                    }
                elif reward_calc.current_tier == 2:
                    # Tier 2: favor army production and upgrades
                    action_weights = {
                        # Barracks units
                        "train_marine": 0.12,
                        "train_marauder": 0.06,
                        "train_reaper": 0.02,
                        "train_ghost": 0.02,
                        
                        # Factory units
                        "train_hellion": 0.03,
                        "train_hellbat": 0.03,
                        "train_siege_tank": 0.05,
                        "train_thor": 0.02,
                        "train_cyclone": 0.02,
                        "train_widow_mine": 0.02,
                        
                        # Starport units
                        "train_viking": 0.03,
                        "train_medivac": 0.04,
                        "train_liberator": 0.03,
                        "train_raven": 0.02,
                        "train_banshee": 0.02,
                        "train_battlecruiser": 0.02,
                        
                        # Research
                        "research_barracks_techlab": 0.04,
                        "research_factory_techlab": 0.02,
                        "research_starport_techlab": 0.02,
                        "research_engineering_bay": 0.03,
                        "research_armory": 0.02,
                        "research_ghost_academy": 0.01,
                        
                        # Maintain economy (lower weight)
                        "train_worker": 0.06,
                        "build_supply_depot": 0.06,
                        "send_idle_workers_to_mine": 0.04,
                        "build_barracks": 0.04,
                        "build_factory": 0.03,
                        "build_starport": 0.03,
                        "call_down_mule": 0.03,
                        
                        # Addons
                        "add_barracks_techlab": 0.02,
                        "add_barracks_reactor": 0.02,
                        "add_factory_techlab": 0.02,
                        "add_starport_techlab": 0.02,
                        
                        "do_nothing": 0.01,
                    }
                else:
                    # Tier 3: favor combat actions
                    action_weights = {
                        # Combat macros
                        "attack_aggressor": 0.12,
                        "attack_siege": 0.08,
                        "attack_support": 0.08,
                        "attack_harass": 0.06,
                        "final_push": 0.08,
                        "retreat_to_base": 0.03,
                        
                        # Keep producing army
                        "train_marine": 0.08,
                        "train_marauder": 0.04,
                        "train_siege_tank": 0.04,
                        "train_medivac": 0.03,
                        "train_viking": 0.02,
                        "train_liberator": 0.02,
                        "train_banshee": 0.02,
                        "train_battlecruiser": 0.02,
                        
                        # Maintain economy (lower weight)
                        "train_worker": 0.04,
                        "build_supply_depot": 0.04,
                        "send_idle_workers_to_mine": 0.03,
                        "call_down_mule": 0.02,
                        
                        # Abilities
                        "scanner_sweep": 0.03,
                        
                        # Production buildings
                        "build_barracks": 0.02,
                        "build_factory": 0.02,
                        "build_starport": 0.02,
                        
                        "do_nothing": 0.01,
                    }
                
                # Tier 4: Balanced weights - network decides strategy
                if reward_calc.current_tier == 4:
                    action_weights = {
                        # Combat - moderate
                        "attack_aggressor": 0.06,
                        "attack_siege": 0.05,
                        "attack_support": 0.05,
                        "attack_harass": 0.04,
                        "final_push": 0.05,
                        "retreat_to_base": 0.02,
                        
                        # All units - balanced
                        "train_marine": 0.06,
                        "train_marauder": 0.03,
                        "train_reaper": 0.01,
                        "train_ghost": 0.01,
                        "train_hellion": 0.02,
                        "train_hellbat": 0.02,
                        "train_siege_tank": 0.03,
                        "train_thor": 0.01,
                        "train_cyclone": 0.01,
                        "train_widow_mine": 0.01,
                        "train_viking": 0.02,
                        "train_medivac": 0.03,
                        "train_liberator": 0.02,
                        "train_raven": 0.01,
                        "train_banshee": 0.02,
                        "train_battlecruiser": 0.01,
                        
                        # Economy - balanced
                        "train_worker": 0.05,
                        "build_supply_depot": 0.04,
                        "send_idle_workers_to_mine": 0.03,
                        "send_idle_workers_to_gas": 0.02,
                        "call_down_mule": 0.02,
                        "build_refinery": 0.02,
                        "build_command_center": 0.02,
                        
                        # Buildings - balanced
                        "build_barracks": 0.03,
                        "build_factory": 0.02,
                        "build_starport": 0.02,
                        "build_engineering_bay": 0.01,
                        "build_armory": 0.01,
                        
                        # Upgrades
                        "upgrade_orbital_command": 0.02,
                        "research_barracks_techlab": 0.02,
                        "research_engineering_bay": 0.01,
                        
                        # Abilities
                        "scanner_sweep": 0.02,
                        
                        "do_nothing": 0.01,
                    }
                
                weighted = [(a, action_weights.get(a, 0.02)) for a in legal_actions]
                actions_list, weights = zip(*weighted)
                weights = np.array(weights) / np.sum(weights)
                action_name = np.random.choice(actions_list, p=weights)
                nx = random.uniform(0, 1)
                ny = random.uniform(0, 1)
            else:
                # Policy action
                request_id = f"{worker_id}_{episode}_{step_count}"
                prediction_queue.put({
                    'command': 'PREDICT',
                    'request_id': request_id,
                    'worker_id': worker_id,
                    'state': state.tolist(),
                    'legal_actions': legal_actions,
                })
                
                result_received = False
                timeout = time_module.time() + 30
                
                while time_module.time() < timeout:
                    try:
                        tmp = result_queue.get(timeout=0.5)
                        if tmp['request_id'] == request_id:
                            action_name = tmp['action']
                            nx = tmp['nx']
                            ny = tmp['ny']
                            log_prob = tmp.get('log_prob', 0.0)
                            # Get legal actions from result if available (for consistency)
                            if 'legal_actions' in tmp:
                                legal_action_indices = tmp['legal_actions']
                            result_received = True
                            break
                        else:
                            try:
                                result_queue.put(tmp, timeout=1)
                            except queue.Full:
                                pass
                    except queue.Empty:
                        continue
                
                if not result_received:
                    action_name = random.choice(legal_actions)
                    nx = ny = 0.0
                    log_prob = 0.0
            
            x_world, y_world = coord_transform.normalized_to_world(nx, ny)
            
            action_fn = ACTION_REGISTRY[action_name]
            
            if action_name == "build_command_center":
                action = action_fn(obs, base_tracker)
            elif ACTION_NEEDS_COORDS.get(action_name, False):
                action = action_fn(obs, x_world, y_world)
            else:
                action = action_fn(obs)

            try:
                obs = env.step([action])[0]
            except ValueError:
                print(f"Worker {worker_id}: Game sync error, skipping step")
                continue
            except Exception as e:
                print(f"Worker {worker_id}: CRASH - {action_name}: {e}")
                with open(f"crash_log_worker_{worker_id}.txt", "a") as f:
                    f.write(f"Ep {episode}, Step {step_count}, {action_name}: {e}\n")
                raise
            
            done = obs.last()
            step_count += 1
            
            # Calculate tiered step reward with placement info
            step_reward = reward_calc.calculate_step_reward(perception, action_name, obs, x_world, y_world)
            step_rewards.append(step_reward)
            
            action_idx = ACTION_TO_INDEX.get(action_name, 0)
            episode_experiences.append({
                'state': state.tolist(),
                'action': action_idx,
                'coords': (nx, ny),
                'reward': 0.0,
                'log_prob': log_prob,
                'legal_actions': legal_action_indices,  # FIXED: Store for training masking
            })
            
            x_viz, y_viz = coord_transform.normalized_to_visualization(nx, ny)
            logger.log_action(action_name, x_viz, y_viz, step_count, perception)
            
            if step_count % 100 == 0:
                logger.log_economy_snapshot(step_count, perception)
            
            perception = GamePerception(obs)
        
        # Episode end
        if obs.reward > 0:
            outcome = 'win'
        elif obs.reward < 0:
            outcome = 'loss'
        else:
            outcome = 'timeout'
        
        # Calculate terminal reward and check graduation
        terminal_reward = reward_calc.calculate_terminal_reward(outcome, step_count)
        graduated, current_tier, tier_score = reward_calc.end_episode(outcome, step_count)
        
        discounted_rewards = compute_discounted_rewards(step_rewards, terminal_reward, gamma=GAMMA)
        
        for exp, reward in zip(episode_experiences, discounted_rewards):
            exp['reward'] = reward
        
        total_reward = tier_score  # Use the actual episode score from reward calculator
        
        if episode_experiences:
            try:
                update_queue.put(episode_experiences, timeout=5)
            except:
                print(f"Worker {worker_id}: Warning - update queue full")
        
        logger.end_episode(outcome, step_count, total_reward, perception, obs)
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        # Print with tier info
        tier_status = reward_calc.get_status()
        print(f"Worker {worker_id} Ep {episode+1}/{episodes_per_worker}: "
              f"{outcome.upper()} ({step_count} steps) | {tier_status} | "
              f"ε={epsilon:.3f}, R={total_reward:.2f}")
    
    print(f"Worker {worker_id} completed at Tier {reward_calc.current_tier}")
    env.close()