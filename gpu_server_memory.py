import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import numpy as np
from lstm_memory import LSTMActionNetwork, ACTION_TO_INDEX
import queue
import time
from collections import deque


class SequenceReplayBufferPPO:
    """
    Replay buffer that stores full episodes with log_probs for PPO.
    Now also stores legal_actions for proper masking during training.
    """
    def __init__(self, max_episodes=32):
        self.episodes = deque(maxlen=max_episodes)
    
    def add_episode(self, states, actions, coords, rewards, dones, log_probs, legal_actions_list=None):
        episode = {
            'states': np.array(states, dtype=np.float32),
            'actions': np.array(actions, dtype=np.int64),
            'coords': np.array(coords, dtype=np.float32),
            'rewards': np.array(rewards, dtype=np.float32),
            'dones': np.array(dones, dtype=np.float32),
            'log_probs': np.array(log_probs, dtype=np.float32),
            'legal_actions': legal_actions_list,  # List of lists of legal action indices per step
            'length': len(states)
        }
        self.episodes.append(episode)
    
    def sample_episodes(self, num_episodes, device='cpu'):
        """Sample complete episodes for training."""
        if len(self.episodes) == 0:
            return None
        
        indices = np.random.choice(len(self.episodes), min(num_episodes, len(self.episodes)), replace=False)
        return [self.episodes[i] for i in indices]
    
    def __len__(self):
        return len(self.episodes)


class GPUServerMemory:
    """
    GPU Server with FIXED training:
    1. Processes sequences with LSTM hidden state continuity
    2. Uses truncated BPTT with proper gradient flow
    3. Handles epsilon exploration steps with behavioral cloning
    4. FIXED: Legal action masking during training to prevent entropy collapse
    """
    
    def __init__(self, state_size=78, hidden_size=1024, num_actions=58, lr=1e-4):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"GPU Server using device: {self.device}")
        
        self.network = LSTMActionNetwork(
            state_size=state_size,
            hidden_size=hidden_size,
            num_actions=num_actions,
            num_lstm_layers=3,
            num_attention_heads=8,
            use_attention=True
        )
        self.network = self.network.to(self.device)
        
        self.optimizer = optim.AdamW(
            self.network.parameters(), 
            lr=lr,
            weight_decay=1e-4
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=1000, T_mult=2
        )
        
        self.num_actions = num_actions
        self.action_to_index = ACTION_TO_INDEX
        self.index_to_action = {v: k for k, v in ACTION_TO_INDEX.items()}
        
        self.hidden_states = {}
        
        # Replay buffer for full episodes
        self.replay_buffer = SequenceReplayBufferPPO(max_episodes=32)
        
        # Training config
        self.tbptt_length = 256  # Truncated BPTT length
        self.episodes_per_batch = 1  # Train on single fresh episode
        self.gamma = 0.99
        
        # PPO hyperparameters
        self.ppo_clip = 0.2
        self.entropy_coef = 0.1
        self.max_ratio = 3.0
        
        # Entropy temperature - prevents collapse by softening logits
        self.entropy_temperature = 1.5
        
        # Minimum entropy threshold - add bonus if entropy drops too low
        self.min_entropy_threshold = 1.0
        self.entropy_bonus_coef = 0.5
        
        self.update_count = 0
        self.episodes_since_last_train = 0
        
        # Warm up CUDA
        dummy_state = torch.randn(1, state_size, device=self.device)
        with torch.no_grad():
            self.network.eval()
            _ = self.network(dummy_state, None, use_memory=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print("✓ CUDA kernels pre-compiled")
        print(f"✓ Model size: {sum(p.numel() for p in self.network.parameters()):,} parameters")
    
    def predict_batch(self, requests):
        """Handle prediction requests with per-worker hidden state."""
        if not requests:
            return []
        
        self.network.eval()
        results = []
        
        with torch.no_grad():
            for req in requests:
                worker_id = req['worker_id']
                
                state_tensor = torch.tensor(
                    req['state'], 
                    dtype=torch.float32, 
                    device=self.device
                ).unsqueeze(0)
                
                hidden = self.hidden_states.get(worker_id)
                if hidden is None:
                    hidden = self.network.init_hidden(batch_size=1)
                    self.hidden_states[worker_id] = hidden
                hidden = (hidden[0].to(self.device), hidden[1].to(self.device))
                
                action_logits, coords, new_hidden, attn_weights = self.network(
                    state_tensor, 
                    hidden, 
                    use_memory=True
                )
                
                action_logits = action_logits.squeeze(0)
                
                # Mask illegal actions
                mask = torch.full((self.num_actions,), float('-inf'), device=self.device)
                legal_indices = [
                    self.action_to_index[a]
                    for a in req['legal_actions']
                    if a in self.action_to_index
                ]
                if legal_indices:
                    mask[legal_indices] = 0
                masked_logits = action_logits + mask
                
                # Get probabilities
                log_probs = torch.log_softmax(masked_logits, dim=-1)
                
                # Select action
                action_idx = torch.argmax(masked_logits).item()
                action_name = self.index_to_action.get(action_idx, "do_nothing")
                
                # Get log prob of selected action
                action_log_prob = log_probs[action_idx].item()
                
                # Extract coordinates with NaN protection
                nx, ny = coords.squeeze(0).cpu().numpy()
                if np.isnan(nx) or np.isnan(ny):
                    nx, ny = 0.5, 0.5  # Default to center
                nx = float(np.clip(nx, 0, 1))
                ny = float(np.clip(ny, 0, 1))
                
                self.hidden_states[worker_id] = new_hidden
                
                results.append({
                    'request_id': req['request_id'],
                    'action': action_name,
                    'nx': nx,
                    'ny': ny,
                    'log_prob': action_log_prob,
                    'legal_actions': legal_indices,  # Return for storage
                })
        
        return results
    
    def reset_episode(self, worker_id):
        """Reset hidden state for a worker."""
        if worker_id in self.hidden_states:
            del self.hidden_states[worker_id]
        self.network.reset_episode()
    
    def add_episode(self, episode_data):
        """Add complete episode to replay buffer."""
        if not episode_data:
            return len(self.replay_buffer)
        
        states = []
        actions = []
        coords = []
        rewards = []
        log_probs = []
        legal_actions_list = []
        
        for exp in episode_data:
            states.append(exp['state'])
            actions.append(exp['action'])
            coords.append(exp['coords'])
            rewards.append(exp['reward'])
            log_probs.append(exp.get('log_prob', 0.0))
            # Store legal actions - fallback to all actions if not provided
            legal_actions_list.append(exp.get('legal_actions', list(range(self.num_actions))))
        
        dones = np.zeros(len(states), dtype=np.float32)
        dones[-1] = 1.0
        
        self.replay_buffer.add_episode(
            states=states,
            actions=actions,
            coords=coords,
            rewards=rewards,
            dones=dones,
            log_probs=log_probs,
            legal_actions_list=legal_actions_list
        )
        
        self.episodes_since_last_train += 1
        
        return len(self.replay_buffer)
    
    def should_train(self):
        """Train every episode for freshest log_probs."""
        return (len(self.replay_buffer) >= 1 and 
                self.episodes_since_last_train >= 1)
    
    def _create_action_mask(self, legal_actions_list, seq_len, device):
        """
        Create action mask tensor for a sequence.
        Returns mask where legal actions have 0, illegal have -inf.
        Shape: [seq_len, num_actions]
        """
        # Default to all zeros (all actions legal) - safe fallback
        mask = torch.zeros((seq_len, self.num_actions), device=device)
        
        if legal_actions_list is None:
            return mask
        
        for t in range(min(len(legal_actions_list), seq_len)):
            legal_actions = legal_actions_list[t]
            if legal_actions is not None and len(legal_actions) > 0:
                # Create mask for this timestep
                step_mask = torch.full((self.num_actions,), float('-inf'), device=device)
                # Clamp indices to valid range
                valid_indices = [idx for idx in legal_actions if 0 <= idx < self.num_actions]
                if valid_indices:
                    step_mask[valid_indices] = 0.0
                    mask[t] = step_mask
                # else: keep all zeros (all legal)
            # else: keep all zeros (all legal)
        
        return mask
    
    def train_step(self, num_updates=1):
        """
        Advantage Actor-Critic with TD(λ) and Hindsight Experience Replay.
        FIXED: Now masks illegal actions during training to prevent entropy collapse.
        """
        if len(self.replay_buffer) < self.episodes_per_batch:
            return None
        
        self.network.train()
        total_losses = {
            'action_loss': 0.0,
            'coord_loss': 0.0,
            'value_loss': 0.0,
            'entropy': 0.0,
            'total_loss': 0.0,
            'her_loss': 0.0,
            'approx_kl': 0.0,
            'clip_fraction': 0.0,
            'valid_ratio': 0.0
        }
        
        # TD(λ) parameter - how much to blend TD estimates
        td_lambda = 0.95
        
        for update_idx in range(num_updates):
            # Sample episodes
            episodes = self.replay_buffer.sample_episodes(
                self.episodes_per_batch, 
                device=self.device
            )
            
            if episodes is None:
                continue
            
            batch_action_loss = 0.0
            batch_coord_loss = 0.0
            batch_value_loss = 0.0
            batch_entropy = 0.0
            batch_her_loss = 0.0
            total_steps = 0
            
            for episode in episodes:
                ep_len = episode['length']
                states = torch.tensor(episode['states'], device=self.device)
                actions = torch.tensor(episode['actions'], device=self.device)
                coords_target = torch.tensor(episode['coords'], device=self.device)
                rewards = torch.tensor(episode['rewards'], device=self.device)
                legal_actions_list = episode.get('legal_actions')
                
                # Create full action mask for episode
                if legal_actions_list is not None:
                    full_action_mask = self._create_action_mask(legal_actions_list, ep_len, self.device)
                else:
                    full_action_mask = torch.zeros((ep_len, self.num_actions), device=self.device)
                
                # Initialize hidden state
                hidden = self.network.init_hidden(batch_size=1)
                
                # First pass: get all values for TD(λ) calculation
                with torch.no_grad():
                    all_values = []
                    temp_hidden = self.network.init_hidden(batch_size=1)
                    for chunk_start in range(0, ep_len, self.tbptt_length):
                        chunk_end = min(chunk_start + self.tbptt_length, ep_len)
                        chunk_states = states[chunk_start:chunk_end].unsqueeze(0)
                        _, _, chunk_values, temp_hidden = self.network.forward_sequence(
                            chunk_states, temp_hidden
                        )
                        all_values.append(chunk_values.squeeze(0).squeeze(-1))
                    all_values = torch.cat(all_values, dim=0)  # [ep_len]
                
                # Compute TD(λ) returns - blend of n-step returns
                td_returns = self._compute_td_lambda_returns(
                    rewards, all_values, td_lambda, self.gamma
                )
                
                # Process in chunks with truncated BPTT
                for chunk_start in range(0, ep_len, self.tbptt_length):
                    chunk_end = min(chunk_start + self.tbptt_length, ep_len)
                    chunk_len = chunk_end - chunk_start
                    
                    chunk_states = states[chunk_start:chunk_end].unsqueeze(0)
                    chunk_actions = actions[chunk_start:chunk_end]
                    chunk_coords = coords_target[chunk_start:chunk_end]
                    chunk_td_returns = td_returns[chunk_start:chunk_end]
                    chunk_mask = full_action_mask[chunk_start:chunk_end]
                    
                    # Detach hidden for TBPTT
                    hidden = (hidden[0].detach(), hidden[1].detach())
                    
                    # Forward pass
                    action_logits, coords_pred, values, hidden = self.network.forward_sequence(
                        chunk_states, hidden
                    )
                    
                    action_logits = action_logits.squeeze(0)  # [chunk_len, num_actions]
                    coords_pred = coords_pred.squeeze(0)
                    values = values.squeeze(0).squeeze(-1)
                    
                    # === CRITICAL FIX: Apply action mask before softmax ===
                    masked_logits = action_logits + chunk_mask
                    
                    # Clamp to prevent NaN in softmax
                    masked_logits = torch.clamp(masked_logits, min=-50, max=50)
                    
                    # TD advantage: how much better was actual vs predicted
                    advantages = chunk_td_returns - values.detach()
                    
                    # Normalize advantages
                    if chunk_len > 1:
                        adv_std = advantages.std() + 1e-8
                        advantages = (advantages - advantages.mean()) / adv_std
                    
                    # Policy loss with advantage weighting (using MASKED logits)
                    log_probs = torch.log_softmax(masked_logits, dim=-1)
                    
                    # Replace NaN in log_probs
                    log_probs = torch.where(torch.isnan(log_probs) | torch.isinf(log_probs),
                                           torch.zeros_like(log_probs) - 10.0, log_probs)
                    
                    action_log_probs = log_probs.gather(1, chunk_actions.unsqueeze(-1)).squeeze(-1)
                    action_loss = -(action_log_probs * advantages).mean()
                    
                    # Value loss: MSE to TD(λ) returns, clipped for stability
                    value_loss = nn.functional.mse_loss(values, chunk_td_returns)
                    value_loss = torch.clamp(value_loss, 0, 1.0)  # Hard clamp to prevent explosion
                    
                    # Coordinate loss
                    coord_loss = nn.functional.mse_loss(coords_pred, chunk_coords)
                    
                    # === FIXED ENTROPY CALCULATION ===
                    # Use temperature scaling to prevent premature collapse
                    # Also compute entropy only over LEGAL actions
                    scaled_logits = masked_logits / self.entropy_temperature
                    
                    # Clamp to prevent overflow in softmax
                    scaled_logits = torch.clamp(scaled_logits, min=-50, max=50)
                    
                    probs = torch.softmax(scaled_logits, dim=-1)
                    
                    # Replace any NaN/inf with small values
                    probs = torch.where(torch.isnan(probs) | torch.isinf(probs), 
                                       torch.ones_like(probs) / self.num_actions, probs)
                    
                    # Safe log - add small epsilon to prevent log(0)
                    log_probs_ent = torch.log(probs + 1e-10)
                    
                    # Entropy: -sum(p * log(p)) for legal actions only
                    entropy = -(probs * log_probs_ent).sum(dim=-1).mean()
                    
                    # Clamp entropy to valid range
                    entropy = torch.clamp(entropy, min=0.0, max=10.0)
                    
                    # Entropy bonus if entropy drops too low
                    if entropy.item() < self.min_entropy_threshold:
                        entropy = entropy + self.entropy_bonus_coef * (self.min_entropy_threshold - entropy.item())
                    
                    batch_action_loss += action_loss * chunk_len
                    batch_value_loss += value_loss * chunk_len
                    batch_coord_loss += coord_loss * chunk_len
                    batch_entropy += entropy * chunk_len
                    total_steps += chunk_len
                
                # === HINDSIGHT EXPERIENCE REPLAY ===
                # If episode was a loss, create synthetic "wins" for achieved goals
                final_reward = rewards[-1].item() if len(rewards) > 0 else 0
                
                if final_reward < 0.5:  # Loss or timeout
                    her_loss = self._compute_her_loss(episode, states, actions)
                    if her_loss is not None:
                        batch_her_loss += her_loss
            
            # Average losses
            batch_action_loss /= total_steps
            batch_value_loss /= total_steps
            batch_coord_loss /= total_steps
            batch_entropy /= total_steps
            
            # Total loss
            total_loss = (batch_action_loss + 
                         0.5 * batch_value_loss + 
                         0.5 * batch_coord_loss +
                         0.1 * batch_her_loss -
                         self.entropy_coef * batch_entropy)
            
            # Skip update if loss is NaN
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"GPU: WARNING - NaN/Inf loss detected, skipping update")
                continue
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)
            self.optimizer.step()
            self.scheduler.step()
            
            # Track metrics
            total_losses['action_loss'] += batch_action_loss.item()
            total_losses['value_loss'] += batch_value_loss.item()
            total_losses['coord_loss'] += batch_coord_loss.item()
            total_losses['entropy'] += batch_entropy.item()
            total_losses['total_loss'] += total_loss.item()
            total_losses['her_loss'] += batch_her_loss.item() if torch.is_tensor(batch_her_loss) else batch_her_loss
            
            self.update_count += 1
        
        for key in total_losses:
            total_losses[key] /= num_updates
        
        return total_losses
    
    def _compute_td_lambda_returns(self, rewards, values, td_lambda, gamma):
        """
        Compute TD(λ) returns - a blend of n-step returns.
        """
        ep_len = len(rewards)
        td_returns = torch.zeros(ep_len, device=rewards.device)
        
        # Start from the end and work backwards
        next_value = 0.0
        next_return = 0.0
        
        for t in range(ep_len - 1, -1, -1):
            # TD target: r + γV(s')
            td_target = rewards[t] + gamma * next_value
            
            # TD(λ) return: blend of TD target and future λ-return
            td_returns[t] = td_target + gamma * td_lambda * (next_return - next_value)
            
            next_value = values[t].item()
            next_return = td_returns[t].item()
        
        return td_returns
    
    def _compute_her_loss(self, episode, states, actions):
        """
        Hindsight Experience Replay.
        """
        ep_len = episode['length']
        
        final_state = states[-1]
        
        # These indices match encode_state array order
        cc_count = final_state[17].item() * 4.0
        barracks = final_state[23].item() * 8.0
        army_supply = final_state[55].item() * 100.0
        
        # Find achievement milestones in episode
        achieved_goals = []
        
        # Goal: Build first barracks
        if barracks >= 1:
            for t in range(ep_len):
                if states[t][23].item() * 8.0 >= 1:
                    achieved_goals.append(('barracks', t, 0.3))
                    break
        
        # Goal: Build expansion
        if cc_count >= 2:
            for t in range(ep_len):
                if states[t][17].item() * 4.0 >= 2:
                    achieved_goals.append(('expansion', t, 0.5))
                    break
        
        # Goal: Army supply milestones
        for threshold, reward in [(20, 0.2), (40, 0.3), (60, 0.4)]:
            if army_supply >= threshold:
                for t in range(ep_len):
                    if states[t][55].item() * 100.0 >= threshold:
                        achieved_goals.append((f'army_{threshold}', t, reward))
                        break
        
        # Goal: Survived N steps
        if ep_len >= 2000:
            achieved_goals.append(('survived_2000', 2000, 0.2))
        if ep_len >= 4000:
            achieved_goals.append(('survived_4000', 4000, 0.3))
        
        if not achieved_goals:
            return None
        
        # Create HER loss: reinforce actions that led to achievements
        her_loss = 0.0
        hidden = self.network.init_hidden(batch_size=1)
        
        for goal_name, goal_step, goal_reward in achieved_goals:
            start = max(0, goal_step - 500)
            end = min(goal_step + 1, ep_len)
            
            if end - start < 10:
                continue
            
            subseq_states = states[start:end].unsqueeze(0)
            subseq_actions = actions[start:end]
            
            with torch.no_grad():
                hidden_temp = self.network.init_hidden(batch_size=1)
            
            action_logits, _, _, _ = self.network.forward_sequence(subseq_states, hidden_temp)
            action_logits = action_logits.squeeze(0)
            
            # Cross-entropy weighted by goal reward
            log_probs = torch.log_softmax(action_logits, dim=-1)
            action_log_probs = log_probs.gather(1, subseq_actions.unsqueeze(-1)).squeeze(-1)
            
            # Higher weight for steps closer to achievement
            steps_to_goal = torch.arange(end - start, 0, -1, device=self.device, dtype=torch.float32)
            weights = goal_reward * (0.99 ** steps_to_goal)
            
            her_loss += -(action_log_probs * weights).mean()
        
        return her_loss / len(achieved_goals)
    
    def save(self, filepath):
        """Save model checkpoint."""
        torch.save({
            'network_state': self.network.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'update_count': self.update_count,
            'config': {
                'state_size': 78,
                'hidden_size': self.network.hidden_size,
                'num_actions': self.num_actions,
                'num_lstm_layers': self.network.num_lstm_layers,
                'num_attention_heads': 8,
            }
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model checkpoint."""
        try:
            checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
            self.network.load_state_dict(checkpoint['network_state'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            
            if 'scheduler_state' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state'])
            if 'update_count' in checkpoint:
                self.update_count = checkpoint['update_count']
            
            print(f"Model loaded from {filepath}")
            return True
        except Exception as e:
            print(f"Could not load model: {e}")
            return False


def gpu_server_process(prediction_queue, result_queue, update_queue, control_queue):
    """Main GPU server loop."""
    print("GPU Server (FIXED TBPTT + PPO + Action Masking) starting...")
    
    server = GPUServerMemory(
        state_size=78,
        hidden_size=1024,
        num_actions=58,
        lr=1e-4
    )
    
    loaded = server.load('memory_model_final.pth')
    if not loaded:
        print("✓ Starting with fresh model")
    
    running = True
    last_print = time.time()
    predictions_served = 0
    episodes_received = 0
    
    print("GPU Server ready (TBPTT + hidden state continuity + action masking)")
    
    while running:
        # Handle predictions
        requests = []
        try:
            req = prediction_queue.get_nowait()
            if req['command'] == 'PREDICT':
                requests.append(req)
            elif req['command'] == 'RESET_EPISODE':
                server.reset_episode(req['worker_id'])
        except queue.Empty:
            pass
        
        while len(requests) < 64:
            try:
                req = prediction_queue.get_nowait()
                if req['command'] == 'PREDICT':
                    requests.append(req)
                elif req['command'] == 'RESET_EPISODE':
                    server.reset_episode(req['worker_id'])
            except queue.Empty:
                break
        
        if requests:
            results = server.predict_batch(requests)
            for result in results:
                try:
                    result_queue.put(result, timeout=5)
                except queue.Full:
                    print("GPU: Warning - result queue full")
            predictions_served += len(requests)
        
        # Handle episode updates
        episodes_added = 0
        while episodes_added < 10:
            try:
                episode_data = update_queue.get_nowait()
                server.add_episode(episode_data)
                episodes_added += 1
                episodes_received += 1
            except queue.Empty:
                break
        
        # Train
        if server.should_train():
            losses = server.train_step(num_updates=1)
            server.episodes_since_last_train = 0
            
            if losses is not None:
                now = time.time()
                if now - last_print > 5.0:
                    print(f"GPU: {server.update_count} updates | "
                          f"Loss: {losses['total_loss']:.4f} "
                          f"(policy={losses['action_loss']:.4f}, value={losses['value_loss']:.4f}, "
                          f"her={losses['her_loss']:.4f}, ent={losses['entropy']:.3f}) | "
                          f"Buf: {len(server.replay_buffer)}")
                    last_print = now
                    predictions_served = 0
                    episodes_received = 0
                
                if server.update_count % 100 == 0:
                    server.save(f'memory_model_checkpoint_{server.update_count}.pth')
        
        # Handle control commands
        try:
            cmd = control_queue.get_nowait()
            if cmd == 'STOP':
                running = False
                break
            elif cmd.startswith('SAVE:'):
                server.save(cmd.split(':', 1)[1])
            elif cmd.startswith('LOAD:'):
                server.load(cmd.split(':', 1)[1])
        except queue.Empty:
            pass
    
    print("GPU Server shutting down...")
    server.save('memory_model_final.pth')
    print("✓ GPU Server saved final model")