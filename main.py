import time
import random
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units 
from terran_functions import ACTION_REGISTRY, ACTION_NEEDS_COORDS
from terran_legal_actions_logic import get_legal_actions
from lstm_action_network import LSTMActionNetwork, ACTION_TO_INDEX
from absl import flags
from game_perception import GamePerception
import numpy as np
from metrics_logger import MetricsLogger

flags.FLAGS(['main.py'])  

MAP_SIZE = 64     

env = sc2_env.SC2Env(
    map_name="Simple64",
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
    visualize=False
)
network = LSTMActionNetwork(state_size=35, hidden_size=128, num_actions=len(ACTION_TO_INDEX), learning_rate=0.01)
logger = MetricsLogger('training_metrics.jsonl')
epsilon_start = 1.0
epsilon_end = 0.05
epsilon_decay = 0.995
epsilon = epsilon_start
try:
    network.load('terran_model.pkl')
except:
    print("No existing model found, starting fresh")
def producer_count(perception, action_name):
    producer = perception.producer_unit_type(action_name)
    if producer is None:
        return 0
    if isinstance(producer, list):
        return sum(perception.count(p) for p in producer)
    return perception.count(producer)
try:
    for episode in range(1000):
        obs = env.reset()[0]
        network.reset_hidden_state()
        start_corner = GamePerception(obs).get_start_corner()
        last_obs = obs
        logger.start_episode(episode + 1)
        done = False
        step_count = 0
        total_reward = 0
        history = []
        print(f"\n{'='*60}")
        print(f"Episode {episode + 1}")
        print(f"{'='*60}")
        while not done:
            perception = GamePerception(obs)
            state = network.encode_state(perception)
            base_x, base_y = perception.get_base_location()
            base_x_norm = base_x / 63
            base_y_norm = base_y / 63
            
            # Economy snapshot BEFORE choosing action (this is fine here)
            if step_count % 100 == 0:
                logger.log_economy_snapshot(step_count, perception)
            
            legal_actions = get_legal_actions(obs)
            legal_actions = [a for a in legal_actions if a in ACTION_REGISTRY]
            if not legal_actions:
                legal_actions = ["do_nothing"]
            
            if random.random() < epsilon:
                action_fn_name = random.choice(legal_actions)
                if ACTION_NEEDS_COORDS.get(action_fn_name, False):
                    x_norm = random.random()
                    y_norm = random.random()
                else:
                    x_norm = y_norm = 0  
            else:
                action_fn_name, x_norm, y_norm = network.predict(state, legal_actions, ACTION_TO_INDEX)
            
            # LOG ACTION HERE - after action_fn_name is defined
            logger.log_action(action_fn_name, x_norm, y_norm, step_count, perception)
            
            action_fn = ACTION_REGISTRY[action_fn_name]
            if ACTION_NEEDS_COORDS.get(action_fn_name, False):
                SCREEN_WIDTH = 84
                SCREEN_HEIGHT = 84
                x = int(x_norm * (SCREEN_WIDTH - 1))
                y = int(y_norm * (SCREEN_HEIGHT - 1))
                action = action_fn(obs, x, y)
            else:
                action = action_fn(obs)
            try:
                obs = env.step([action])[0]
                done = obs.last()
                step_count += 1
                action_index = ACTION_TO_INDEX.get(action_fn_name, 0)
                history.append((state, action_index, x_norm, y_norm))
            except ValueError as e:
                print(f"Action failed: {action_fn_name} - {e}")
                obs = last_obs
                done = obs.last()
            last_obs = obs
        
        game_loops = obs.observation.game_loop
        game_time_seconds = game_loops / 22.4
        if done: 
            my_ccs = sum(1 for u in obs.observation.raw_units 
                        if u.alliance == features.PlayerRelative.SELF and
                        u.unit_type in [units.Terran.CommandCenter, units.Terran.CommandCenterFlying])
            enemy_ccs = sum(1 for u in obs.observation.raw_units 
                            if u.alliance == features.PlayerRelative.ENEMY and
                            u.unit_type in [units.Terran.CommandCenter, units.Terran.CommandCenterFlying,
                                        units.Zerg.Hatchery, units.Zerg.Lair, units.Zerg.Hive,
                                        units.Protoss.Nexus])
            if my_ccs > 0 and enemy_ccs == 0:
                total_reward = 100
                outcome = 'win'
            elif my_ccs == 0:
                total_reward = -100
                outcome = 'loss'
            else:
                total_reward = 0
                outcome = 'draw'
        else:
            total_reward = 0
            outcome = 'ongoing'
        
        network.update_game_outcome(history, outcome, game_time_seconds)
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        logger.end_episode(outcome, step_count, total_reward, perception, obs)
        
        print(f"\nEpisode {episode + 1} Complete:")
        print(f"  Total Steps: {step_count}")
        print(f"  Total Reward: {float(total_reward):.2f}")
        print(f"  Avg Reward/Step: {float(total_reward)/step_count if step_count > 0 else 0:.4f}")
        print(f"Epsilon after episode {episode + 1}: {epsilon:.4f}")
        
        if (episode + 1) % 50 == 0:
            network.save('terran_model.pkl')
finally:
    env.close()
    network.save('terran_model_final.pkl')
    print("\nTraining complete!")