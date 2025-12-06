from absl import flags
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
FLAGS = flags.FLAGS
FLAGS(sys.argv)
ACTIONS = [
    'do_nothing',
    'build_supply_depot',
    'build_barracks',
    'train_scv',
    'train_marine',
    'attack_enemy_base',
]
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.screen_conv = nn.Sequential(
            nn.Conv2d(17, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.minimap_conv = nn.Sequential(
            nn.Conv2d(7, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4 + 64 * 4 * 4 + 10, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, len(ACTIONS))
        )
    def forward(self, screen, minimap, player):
        screen_out = self.screen_conv(screen)
        screen_flat = screen_out.view(screen_out.size(0), -1)
        minimap_out = self.minimap_conv(minimap)
        minimap_flat = minimap_out.view(minimap_out.size(0), -1)
        combined = torch.cat([screen_flat, minimap_flat, player], dim=1)
        return self.fc(combined)
class ActionTracker:
    def __init__(self):
        self.prev_minerals = 50
        self.prev_units_in_progress = 0
        self.supply_depots_started = 0
        self.barracks_started = 0
        self.scvs_started = 0
        self.marines_started = 0
    def reset(self):
        self.prev_minerals = 50
        self.prev_units_in_progress = 0
        self.supply_depots_started = 0
        self.barracks_started = 0
        self.scvs_started = 0
        self.marines_started = 0
    def check_action_success(self, obs, action_name):
        player = obs.observation.player
        reward = 0.0
        mineral_spent = self.prev_minerals - player.minerals
        if action_name == 'build_supply_depot':
            if mineral_spent >= 95:
                reward = 10.0
                self.supply_depots_started += 1
                print(f"    ✅ Supply depot STARTED! (-100 minerals) Total: {self.supply_depots_started}")
            elif mineral_spent > 0:
                print(f"    ⚠️ Partial mineral spent: {mineral_spent}")
        elif action_name == 'build_barracks':
            if mineral_spent >= 145:
                reward = 15.0
                self.barracks_started += 1
                print(f"    ✅ Barracks STARTED! (-150 minerals) Total: {self.barracks_started}")
            elif mineral_spent > 0:
                print(f"    ⚠️ Partial mineral spent: {mineral_spent}")
        elif action_name == 'train_scv':
            if mineral_spent >= 45:
                reward = 3.0
                self.scvs_started += 1
                print(f"    ✅ SCV training STARTED! (-50 minerals) Total: {self.scvs_started}")
        elif action_name == 'train_marine':
            if mineral_spent >= 45:
                reward = 5.0
                self.marines_started += 1
                print(f"    ✅ Marine training STARTED! (-50 minerals) Total: {self.marines_started}")
        if reward == 0.0 and action_name != 'do_nothing' and action_name != 'attack_enemy_base':
            reward = -1.0
#            print(f"    ❌ Action FAILED: {action_name} (no minerals spent)")
        self.prev_minerals = player.minerals
        return reward
class CombatTracker:
    def __init__(self):
        self.prev_score = 0
        self.prev_army_count = 0
    def reset(self):
        self.prev_score = 0
        self.prev_army_count = 0
    def get_combat_reward(self, obs):
        player = obs.observation.player
        reward = 0.0
        score_diff = obs.observation.score_cumulative.killed_value_units - self.prev_score
        if score_diff > 0:
            reward += score_diff / 3.0
            print(f"    💥 Enemy killed! (+{score_diff/3.0:.1f})")
        army_increase = player.army_count - self.prev_army_count
        if army_increase > 0:
            reward += 2.0 * army_increase
            print(f"    🪖 Army grew: {player.army_count} (+{2.0 * army_increase:.1f})")
        self.prev_score = obs.observation.score_cumulative.killed_value_units
        self.prev_army_count = player.army_count
        return reward
class SmartExecutor:
    def __init__(self):
        self.action_queue = []
        self.last_action = "do_nothing"
        self.cc_location = None
        self.barracks_locations = []
        self.used_locations = []
    def clamp_point(self, pt, obs):
        h = obs.observation.feature_screen.player_relative.shape[0]
        w = obs.observation.feature_screen.player_relative.shape[1]
        return [max(0, min(pt[0], w - 1)),
                max(0, min(pt[1], h - 1))]
    def get_base_location(self, obs):
        if hasattr(obs.observation, "feature_units"):
            for u in obs.observation.feature_units:
                if u.unit_type == 18 and u.alliance == 1:
                    return [int(u.x), int(u.y)]
        pr = obs.observation.feature_screen.player_relative
        y, x = (pr == 1).nonzero()
        if len(x) > 0:
            return [int(x.mean()), int(y.mean())]
        return [40, 40]
    def find_valid_build_location(self, obs):
        base = self.get_base_location(obs)
        bx, by = base
        buildable = obs.observation.feature_screen.buildable
        H, W = buildable.shape
        for d in range(4, 35):
            for angle_i in range(8):
                theta = angle_i * (np.pi / 4)
                x = int(bx + d * np.cos(theta))
                y = int(by + d * np.sin(theta))
                if 0 <= x < W and 0 <= y < H:
                    if buildable[y, x] == 1:
                        ok = True
                        for ux, uy in self.used_locations:
                            if np.hypot(x - ux, y - uy) < 4:
                                ok = False
                                break
                        if ok:
                            return [x, y]
        return self.clamp_point([bx + 6, by + 6], obs)
    def can_afford(self, player, cost):
        return player.minerals >= cost
    def queue_action(self, action_name, obs):
        player = obs.observation.player
        self.last_action = action_name
        if action_name == "build_supply_depot":
            if self.can_afford(player, 100):
                self.action_queue = ["select_worker", "build_supply"]
                return True
        if action_name == "build_barracks":
            if self.can_afford(player, 150) and player.food_cap >= 14:
                self.action_queue = ["select_worker", "build_barracks"]
                return True
        if action_name == "train_scv":
            if self.can_afford(player, 50) and player.food_used < player.food_cap:
                self.action_queue = ["select_cc", "train_scv"]
                return True
        if action_name == "train_marine":
            if self.can_afford(player, 50):
                self.action_queue = ["select_barracks", "train_marine"]
                return True
        if action_name == "attack_enemy_base":
            if player.army_count >= 12 and len(self.barracks_locations) >= 1:
                self.action_queue = ["select_army", "attack"]
                return True
        return False
    def execute_next(self, obs):
        if not self.action_queue:
            return actions.FUNCTIONS.no_op(), False
        step = self.action_queue.pop(0)
        avail = obs.observation.available_actions
        is_final = (len(self.action_queue) == 0)
        if step == "select_worker":
            if actions.FUNCTIONS.select_idle_worker.id in avail:
                return actions.FUNCTIONS.select_idle_worker("select"), False
            if hasattr(obs.observation, "feature_units"):
                for u in obs.observation.feature_units:
                    if u.unit_type == units.Terran.SCV and u.alliance == 1:
                        pt = self.clamp_point([int(u.x), int(u.y)], obs)
                        if actions.FUNCTIONS.select_point.id in avail:
                            return actions.FUNCTIONS.select_point("select", pt), False
            return actions.FUNCTIONS.no_op(), False
        if step == "build_supply" and actions.FUNCTIONS.Build_SupplyDepot_screen.id in avail:
            loc = self.clamp_point(self.find_valid_build_location(obs), obs)
            self.used_locations.append(loc)
            if obs.observation.player.food_cap - obs.observation.player.food_used >= 5:
                return actions.FUNCTIONS.no_op(), is_final
            return actions.FUNCTIONS.Build_SupplyDepot_screen("now", loc), is_final
        if step == "build_barracks" and actions.FUNCTIONS.Build_Barracks_screen.id in avail:
            loc = self.clamp_point(self.find_valid_build_location(obs), obs)
            self.used_locations.append(loc)
            self.barracks_locations.append(loc)
            self._pending_build = "barracks"
            return actions.FUNCTIONS.Build_Barracks_screen("now", loc), is_final
        if step == "select_cc" and actions.FUNCTIONS.select_point.id in avail:
            if not self.cc_location:
                self.cc_location = self.get_base_location(obs)
            pt = self.clamp_point(self.cc_location, obs)
            return actions.FUNCTIONS.select_point("select_all_type", pt), False
        if step == "train_scv" and actions.FUNCTIONS.Train_SCV_quick.id in avail:
            num_cc = sum(1 for u in obs.observation.feature_units
                        if u.unit_type == units.Terran.CommandCenter and u.alliance == 1)
            max_scvs = 16 * num_cc
            if obs.observation.player.food_used < max_scvs:
                return actions.FUNCTIONS.Train_SCV_quick("now"), is_final
            else:
                return actions.FUNCTIONS.no_op(), is_final
        if step == "select_barracks" and actions.FUNCTIONS.select_point.id in avail:
            real = None
            if hasattr(obs.observation, "feature_units"):
                for u in obs.observation.feature_units:
                    if u.unit_type == units.Terran.Barracks and u.alliance == 1:
                        real = [int(u.x), int(u.y)]
                        break
            loc = real if real else (self.barracks_locations[-1] if self.barracks_locations else None)
            if loc:
                return actions.FUNCTIONS.select_point("select_all_type",
                                                    self.clamp_point(loc, obs)), False
            return actions.FUNCTIONS.no_op(), False
        if step == "train_marine" and actions.FUNCTIONS.Train_Marine_quick.id in avail:
            return actions.FUNCTIONS.Train_Marine_quick("now"), is_final
        if step == "select_army" and actions.FUNCTIONS.select_army.id in avail:
            return actions.FUNCTIONS.select_army("select"), False
        if step == "attack" and actions.FUNCTIONS.Attack_minimap.id in avail:
            rel = obs.observation.feature_minimap.player_relative
            ey, ex = (rel == features.PlayerRelative.ENEMY).nonzero()
            if len(ex) > 0:
                pt = [int(ex.mean()), int(ey.mean())]
            else:
                pt = [32, 32]
            return actions.FUNCTIONS.Attack_minimap("now", pt), is_final
        return actions.FUNCTIONS.no_op(), False
class SC2Agent:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu', checkpoint_path=None):
        self.device = device
        self.policy_net = DQN().to(device)
        self.target_net = DQN().to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-4)
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.episodes_trained = 0
        self.executor = SmartExecutor()
        self.action_tracker = ActionTracker()
        self.combat_tracker = CombatTracker()
        print(f"Device: {device}")
        if device == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
    def preprocess_obs(self, obs):
        screen = obs.observation.feature_screen
        screen_data = np.stack([
            screen.height_map, screen.visibility_map, screen.creep, screen.power,
            screen.player_id, screen.player_relative, screen.unit_type, screen.selected,
            screen.unit_hit_points, screen.unit_hit_points_ratio, screen.unit_energy,
            screen.unit_energy_ratio, screen.unit_shields, screen.unit_shields_ratio,
            screen.unit_density, screen.unit_density_aa, screen.effects
        ]) / 255.0
        minimap = obs.observation.feature_minimap
        minimap_data = np.stack([
            minimap.height_map, minimap.visibility_map, minimap.creep,
            minimap.camera, minimap.player_id, minimap.player_relative, minimap.selected
        ]) / 255.0
        player = obs.observation.player
        player_data = np.array([
            player.minerals, player.vespene, player.food_used, player.food_cap,
            player.food_army, player.food_workers, player.idle_worker_count,
            player.army_count, player.warp_gate_count, player.larva_count
        ], dtype=np.float32) / 200.0
        return (torch.FloatTensor(screen_data).unsqueeze(0).to(self.device),
                torch.FloatTensor(minimap_data).unsqueeze(0).to(self.device),
                torch.FloatTensor(player_data).unsqueeze(0).to(self.device))
    def select_macro_action(self, obs):
        if random.random() < self.epsilon:
            return random.randint(0, len(ACTIONS) - 1)
        else:
            screen, minimap, player = self.preprocess_obs(obs)
            with torch.no_grad():
                q_values = self.policy_net(screen, minimap, player)
                return q_values.argmax(1).item()
    def store_transition(self, state, action, reward, next_state, done):
        state_np = tuple(s.detach().cpu().numpy().copy() for s in state)
        next_state_np = tuple(s.detach().cpu().numpy().copy() for s in next_state)
        self.memory.append((state_np, action, reward, next_state_np, done))
    def train_step(self):
        if len(self.memory) < self.batch_size:
            return 0.0
        batch = random.sample(self.memory, self.batch_size)
        total_loss = 0
        for state_np, action, reward, next_state_np, done in batch:
            if isinstance(state_np[0], np.ndarray):
                screen = torch.from_numpy(state_np[0]).float().to(self.device)
                minimap = torch.from_numpy(state_np[1]).float().to(self.device)
                player = torch.from_numpy(state_np[2]).float().to(self.device)
            else:
                screen = state_np[0].to(self.device)
                minimap = state_np[1].to(self.device)
                player = state_np[2].to(self.device)
            q_values = self.policy_net(screen, minimap, player)
            q_value = q_values[0, action]
            if done:
                target = reward
            else:
                if isinstance(next_state_np[0], np.ndarray):
                    next_screen = torch.from_numpy(next_state_np[0]).float().to(self.device)
                    next_minimap = torch.from_numpy(next_state_np[1]).float().to(self.device)
                    next_player = torch.from_numpy(next_state_np[2]).float().to(self.device)
                else:
                    next_screen = next_state_np[0].to(self.device)
                    next_minimap = next_state_np[1].to(self.device)
                    next_player = next_state_np[2].to(self.device)
                
                with torch.no_grad():
                    next_q = self.target_net(next_screen, next_minimap, next_player).max(1)[0]
                    target = reward + self.gamma * next_q.item()
            loss = nn.MSELoss()(q_value, torch.tensor(target, dtype=torch.float32).to(self.device))
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item()
            del screen, minimap, player, q_values, q_value, loss
            if not done:
                del next_screen, next_minimap, next_player
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return total_loss / self.batch_size
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    def save_checkpoint(self, filepath):
        checkpoint = {
            'policy_net_state': self.policy_net.state_dict(),
            'target_net_state': self.target_net.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'episodes_trained': self.episodes_trained,
        }
        torch.save(checkpoint, filepath)
        print(f"  💾 Checkpoint saved to {filepath}")
    def load_checkpoint(self, filepath):
        try:
            checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
            self.policy_net.load_state_dict(checkpoint['policy_net_state'])
            self.target_net.load_state_dict(checkpoint['target_net_state'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            self.epsilon = checkpoint['epsilon']
            self.episodes_trained = checkpoint['episodes_trained']
            if 'memory' in checkpoint:
                self.memory = deque(checkpoint['memory'], maxlen=10000)
            print(f"\n✅ Loaded checkpoint: Episode {self.episodes_trained}, ε={self.epsilon:.3f}")
        except FileNotFoundError:
            print(f"⚠️ No checkpoint found, starting fresh")
    def detect_terminal_result(self, obs):
        units_map = obs.observation.feature_units
        enemy_buildings = 0
        my_units = 0
        for u in units_map:
            if u.alliance == 1:
                my_units += 1
            elif u.alliance == 4: 
                if u.unit_type in [18, 21, 20, 317, 322, 324, 325]: 
                    enemy_buildings += 1
        if enemy_buildings == 0 and my_units > 0:
            return 1  
        if my_units == 0:
            return -1  
        return 0  
def train():
    CHECKPOINT_PATH = "sc2_agent_latest.pth"
    LOAD_CHECKPOINT = True
    env = sc2_env.SC2Env(
        map_name="Simple64",
        players=[
            sc2_env.Agent(sc2_env.Race.terran),
            sc2_env.Bot(sc2_env.Race.random, sc2_env.Difficulty.easy)
        ],
        agent_interface_format=features.AgentInterfaceFormat(
            feature_dimensions=features.Dimensions(screen=84, minimap=64),
            use_feature_units=True),
        step_mul=8,
        game_steps_per_episode=0,  
        visualize=False
    )
    agent = SC2Agent(checkpoint_path=CHECKPOINT_PATH if LOAD_CHECKPOINT else None)
    episodes = 500
    starting_episode = agent.episodes_trained
    print(f"\n{'='*60}")
    print(f"Training from episode {starting_episode} to {starting_episode + episodes}")
    print(f"{'='*60}\n")
    wins = 0
    losses = 0
    try:
        for episode in range(episodes):
            terminal_reward = None
            if 'obs' in locals() and obs is not None and len(obs) > 0:
                terminal_reward = agent.detect_terminal_result(obs[0])
            if terminal_reward is not None and terminal_reward != 0:
                done = True
                episode_reward = terminal_reward
                if terminal_reward > 0:
                    wins += 1
                    print(f"  🎉 VICTORY!")
                elif terminal_reward < 0:
                    losses += 1
                    print(f"  💀 DEFEAT")
            actual_episode = starting_episode + episode
            agent.episodes_trained = actual_episode + 1
            obs = env.reset()
            agent.action_tracker.reset()
            agent.combat_tracker.reset()
            agent.executor = SmartExecutor()
            episode_reward = 0
            macro_action_idx = None
            state = None
            for step in range(2000):
                obs_current = obs[0]
                if agent.executor.action_queue:
                    action_obj, is_final = agent.executor.execute_next(obs_current)
                else:
                    state = agent.preprocess_obs(obs_current)
                    macro_action_idx = agent.select_macro_action(obs_current)
                    action_name = ACTIONS[macro_action_idx]                    
                    if agent.executor.queue_action(action_name, obs_current):
                        action_obj, is_final = agent.executor.execute_next(obs_current)
                    else:
                        action_obj = actions.FUNCTIONS.no_op()
                        is_final = True                
                obs = env.step([action_obj])
                done = obs[0].last()                
                if is_final and agent.executor.last_action != 'do_nothing':
                    action_reward = agent.action_tracker.check_action_success(obs[0], agent.executor.last_action)
                    combat_reward = agent.combat_tracker.get_combat_reward(obs[0])
                    reward = action_reward + combat_reward
                    episode_reward += reward
                    next_state = agent.preprocess_obs(obs[0])
                    agent.store_transition(state, macro_action_idx, reward, next_state, done)
                    try:
                        del state, next_state
                    except:
                        pass
                if done:
                    env_reward = obs[0].reward
                    episode_reward += env_reward
                    if env_reward > 0:
                        wins += 1
                        print(f"  🎉 VICTORY!")
                    elif env_reward < 0:
                        losses += 1
                        print(f"  💀 DEFEAT")
                    break
                if step % 4 == 0 and len(agent.memory) >= agent.batch_size:
                    agent.train_step()
                if step % 50 == 0:  
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)            
            if episode % 5 == 0:
                agent.update_target_network()           
            player = obs[0].observation.player
            total_games = wins + losses
            win_rate = (100 * wins / total_games) if total_games > 0 else 0            
            print(f"\n{'='*60}")
            print(f"Episode {actual_episode + 1} | Reward: {episode_reward:.1f} | ε: {agent.epsilon:.3f}")
            print(f"  Built: Depots={agent.action_tracker.supply_depots_started} Barracks={agent.action_tracker.barracks_started}")
            print(f"  Trained: SCVs={agent.action_tracker.scvs_started} Marines={agent.action_tracker.marines_started}")
            print(f"  Final: Supply {player.food_used}/{player.food_cap} | Army={player.army_count} | Minerals={player.minerals}")
            print(f"  Record: {wins}W / {losses}L ({win_rate:.1f}% win rate)")            
            if (episode + 1) % 10 == 0:
                agent.save_checkpoint(CHECKPOINT_PATH)
                if (episode + 1) % 50 == 0:
                    agent.save_checkpoint(f"sc2_agent_ep{actual_episode + 1}.pth")
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted")
    finally:
        env.close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
if __name__ == "__main__":
    train()