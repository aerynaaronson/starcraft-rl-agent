from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from game_perception import GamePerception, encode_state  # Import encode_state from game_perception
from terran_actions import ACTION_REGISTRY, ACTION_NEEDS_COORDS
from terran_legal_actions_logic import get_legal_actions
from lstm_memory import ACTION_TO_INDEX
from metrics_logger import MetricsLogger
from coordinate_transform import CoordinateTransform
from worker_rewards import RewardCalculator, compute_discounted_rewards
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
    Game worker with memory-enabled LSTM support.
    
    Args:
        worker_id: Unique worker identifier
        prediction_queue: Queue for sending prediction requests to GPU server
        result_queue: Queue for receiving predictions from GPU server
        update_queue: Queue for sending episode experiences for training
        episodes_per_worker: Number of episodes this worker should run
        map_name: Name of the map to play on (default: 'Simple64')
    """
    max_steps = 10_000
    epsilon_start = 0.3
    epsilon_end = 0.05
    epsilon_decay = 0.995
    
    print(f"Worker {worker_id} starting (memory-enabled)...")
    print(f"  Map: {map_name}, Max steps: {max_steps}")
    
    logger = MetricsLogger(f'training_metrics_worker_{worker_id}.jsonl')
    env = create_env(worker_id, map_name)
    
    epsilon = epsilon_start
    base_tracker = BaseTracker()
    
    for episode in range(episodes_per_worker):
        base_tracker.reset()
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
        
        reward_calc = RewardCalculator()
        reward_calc.base_tracker = base_tracker  # Share the same instance
        
        logger.start_episode(episode + 1, perception.get_start_corner())
        
        episode_experiences = []
        step_rewards = []
        done = False
        step_count = 0
        search_index = [0]
        last_enemy_buildings = []
        
        while not done and step_count < max_steps:
            if not base_tracker.expansion_centroids:
                base_tracker.initialize_expansions(obs)
            
            # Update base status every step so build_command_center sees current state
            base_tracker.update_base_status(obs)
            
            perception.obs = obs
            perception.update_enemy_info(obs)
            
            # Use 78-feature encode_state with coordinate transform
            state = encode_state(perception, coord_transform)
            
            legal_actions = get_legal_actions(obs)
            legal_actions = [a for a in legal_actions if a in ACTION_REGISTRY]
            if not legal_actions:
                legal_actions = ["do_nothing"]
            
            if random.random() < epsilon:
                action_weights = {
                    "attack_aggressor": 0.10,
                    "attack_siege": 0.05,
                    "attack_support": 0.05,
                    "attack_harass": 0.05,
                    "final_push": 0.05,
                    "train_marine": 0.20,
                    "train_worker": 0.15,
                    "build_barracks": 0.12,
                    "build_supply_depot": 0.10,
                    "send_idle_workers_to_mine": 0.08,
                    "do_nothing": 0.01,
                }
                weighted = [(a, action_weights.get(a, 0.05)) for a in legal_actions]
                actions_list, weights = zip(*weighted)
                weights = np.array(weights) / np.sum(weights)
                action_name = np.random.choice(actions_list, p=weights)
                nx = random.uniform(0, 1)
                ny = random.uniform(0, 1)
            else:
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
                            result_received = True
                            break
                        else:
                            try:
                                result_queue.put(tmp, timeout=1)
                            except queue.Full:
                                pass  # Drop stale result
                    except queue.Empty:
                        continue
                
                if not result_received:
                    action_name = random.choice(legal_actions)
                    nx = ny = 0.0
            
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
            
            step_reward = reward_calc.calculate_step_reward(perception, action_name, obs)
            step_rewards.append(step_reward)
            
            action_idx = ACTION_TO_INDEX.get(action_name, 0)
            episode_experiences.append({
                'state': state.tolist(),
                'action': action_idx,
                'coords': (nx, ny),
                'reward': 0.0,
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
        
        terminal_reward = reward_calc.calculate_terminal_reward(outcome, step_count)
        discounted_rewards = compute_discounted_rewards(step_rewards, terminal_reward, gamma=0.99)
        
        for exp, reward in zip(episode_experiences, discounted_rewards):
            exp['reward'] = reward
        
        total_reward = discounted_rewards[0] if discounted_rewards else terminal_reward
        
#        if episode < 3:
#            print(f"  [DEBUG] steps={len(step_rewards)}, terminal={terminal_reward:.2f}, return={total_reward:.2f}")
        
        if episode_experiences:
            try:
                update_queue.put(episode_experiences, timeout=5)
            except:
                print(f"Worker {worker_id}: Warning - update queue full")
        
        logger.end_episode(outcome, step_count, total_reward, perception, obs)
        reward_calc.reset()
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        print(f"Worker {worker_id} Ep {episode+1}/{episodes_per_worker}: "
              f"{outcome.upper()} ({step_count} steps, Îµ={epsilon:.3f}, R={total_reward:.2f})")
    
    print(f"Worker {worker_id} completed, shutting down...")
    env.close()