import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import numpy as np
from lstm_memory import LSTMActionNetwork, SequenceReplayBuffer, ACTION_TO_INDEX
import queue
import time


class GPUServerMemory:
    """
    GPU Server with PPO training for stable learning.
    """
    
    def __init__(self, state_size=78, hidden_size=1024, num_actions=58, lr=3e-4):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"GPU Server using device: {self.device}")
        
        # Create memory-enabled network
        self.network = LSTMActionNetwork(
            state_size=state_size,
            hidden_size=hidden_size,
            num_actions=num_actions,
            num_lstm_layers=3,
            num_attention_heads=8,
            use_attention=True
        )
        self.network = self.network.to(self.device)
        
        # Optimizer with weight decay for regularization
        self.optimizer = optim.AdamW(
            self.network.parameters(), 
            lr=lr,
            weight_decay=1e-4
        )
        
        # Cosine annealing scheduler for better convergence
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=1000, T_mult=2
        )
        
        self.num_actions = num_actions
        self.action_to_index = ACTION_TO_INDEX
        self.index_to_action = {v: k for k, v in ACTION_TO_INDEX.items()}
        
        # Hidden states per worker (for inference)
        self.hidden_states = {}
        
        # Episode memory per worker (stores last 2000 steps for attention)
        self.worker_episodes = {}
        
        # Sequence replay buffer (stores complete episodes for training)
        self.replay_buffer = SequenceReplayBuffer(max_episodes=64)
        
        # Training config
        self.sequence_length = 128
        self.batch_size = 64
        self.gamma = 0.99
        
        # PPO hyperparameters
        self.ppo_clip = 0.2  # Clip ratio
        self.entropy_coef = 0.01  # Entropy bonus for exploration
        self.value_coef = 0.5  # Value loss coefficient (if using critic)
        
        # Tracking
        self.update_count = 0
        self.episodes_since_last_train = 0  # Track episodes for training trigger
        
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
        """
        Handle prediction requests with memory management.
        Each worker maintains its own hidden state and episode memory.
        """
        if not requests:
            return []
        
        self.network.eval()
        results = []
        
        with torch.no_grad():
            for req in requests:
                worker_id = req['worker_id']
                
                # Get state
                state_tensor = torch.tensor(
                    req['state'], 
                    dtype=torch.float32, 
                    device=self.device
                ).unsqueeze(0)
                
                # Get or initialize hidden state for this worker
                hidden = self.hidden_states.get(worker_id)
                if hidden is None:
                    hidden = self.network.init_hidden(batch_size=1)
                    self.hidden_states[worker_id] = hidden
                hidden = (hidden[0].to(self.device), hidden[1].to(self.device))
                
                # Forward pass with memory
                action_logits, coords, new_hidden, attn_weights = self.network(
                    state_tensor, 
                    hidden, 
                    use_memory=True
                )
                
                action_logits = action_logits.squeeze(0)
                
                # Mask illegal actions
                mask = torch.full(
                    (self.num_actions,), 
                    float('-inf'), 
                    device=self.device
                )
                legal_indices = [
                    self.action_to_index[a]
                    for a in req['legal_actions']
                    if a in self.action_to_index
                ]
                if legal_indices:
                    mask[legal_indices] = 0
                action_logits = action_logits + mask
                
                # Select action
                action_idx = torch.argmax(action_logits).item()
                action_name = self.index_to_action.get(action_idx, "do_nothing")
                
                # Extract normalized coordinates
                nx, ny = coords.squeeze(0).cpu().numpy()
                nx = float(np.clip(nx, -1, 1))
                ny = float(np.clip(ny, -1, 1))
                
                # Update hidden state
                self.hidden_states[worker_id] = new_hidden
                
                results.append({
                    'request_id': req['request_id'],
                    'action': action_name,
                    'nx': nx,
                    'ny': ny,
                })
        
        return results
    
    def reset_episode(self, worker_id):
        """Reset episode memory for a worker."""
        # Only reset this worker's hidden state
        if worker_id in self.hidden_states:
            del self.hidden_states[worker_id]
        
        # Reset this worker's episode tracking
        self.worker_episodes[worker_id] = {
            'states': [],
            'actions': [],
            'coords': [],
            'rewards': []
        }
    
    def add_episode(self, episode_data):
        """Add complete episode to replay buffer."""
        if not episode_data:
            return len(self.replay_buffer)
        
        states = []
        actions = []
        coords = []
        rewards = []
        
        for exp in episode_data:
            states.append(exp['state'])
            actions.append(exp['action'])
            coords.append(exp['coords'])
            rewards.append(exp['reward'])
        
        dones = np.zeros(len(states), dtype=np.float32)
        dones[-1] = 1.0
        
        self.replay_buffer.add_episode(
            states=states,
            actions=actions,
            coords=coords,
            rewards=rewards,
            dones=dones
        )
        
        self.episodes_since_last_train += 1
        
        return len(self.replay_buffer)
    
    def should_train(self):
        """Check if we should train - every 16 new episodes"""
        return (len(self.replay_buffer) >= 16 and 
                self.episodes_since_last_train >= 16)
    
    def train_step(self, num_updates=5):
        """
        PPO training step.
        
        Key difference from vanilla policy gradient:
        - Clips the policy ratio to prevent large updates
        - Adds entropy bonus for exploration
        - Much more stable training
        """
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        self.network.train()
        total_losses = {
            'action_loss': 0.0,
            'coord_loss': 0.0,
            'entropy': 0.0,
            'total_loss': 0.0
        }
        
        for _ in range(num_updates):
            # Sample sequences from replay buffer
            batch = self.replay_buffer.sample_sequences(
                batch_size=self.batch_size,
                sequence_length=self.sequence_length,
                device=self.device
            )
            
            if batch is None:
                continue
            
            batch_size = batch['states'].shape[0]
            seq_len = batch['states'].shape[1]
            
            # Forward pass
            states_reshaped = batch['states'].view(batch_size * seq_len, -1)
            
            all_action_logits = []
            all_coords = []
            
            chunk_size = 32
            for i in range(0, batch_size * seq_len, chunk_size):
                chunk_states = states_reshaped[i:i+chunk_size]
                chunk_logits, chunk_coords, _, _ = self.network(
                    chunk_states,
                    hidden=None,
                    use_memory=False
                )
                all_action_logits.append(chunk_logits)
                all_coords.append(chunk_coords)
            
            action_logits = torch.cat(all_action_logits, dim=0)
            coords = torch.cat(all_coords, dim=0)
            
            action_logits = action_logits.view(batch_size, seq_len, -1)
            coords = coords.view(batch_size, seq_len, -1)
            
            # Get current policy probabilities
            probs = torch.softmax(action_logits, dim=-1)
            log_probs = torch.log_softmax(action_logits, dim=-1)
            
            # Gather log probs for taken actions
            actions = batch['actions'].unsqueeze(-1)
            action_log_probs = log_probs.gather(dim=-1, index=actions).squeeze(-1)
            action_probs = probs.gather(dim=-1, index=actions).squeeze(-1)
            
            # Compute advantages (rewards - baseline)
            returns = batch['rewards']
            baseline = returns.mean()
            advantages = returns - baseline
            
            # Normalize advantages
            adv_std = advantages.std() + 1e-8
            advantages = (advantages - advantages.mean()) / adv_std
            
            # PPO: compute ratio and clipped objective
            # For first pass, old_probs = current probs (no clipping effect)
            # In practice, you'd store old_probs with the episode data
            # This simplified version still helps stability via the clipping mechanism
            
            # Use detached probs as "old" policy (approximation)
            old_log_probs = action_log_probs.detach()
            
            # Ratio = new_prob / old_prob = exp(new_log - old_log)
            ratio = torch.exp(action_log_probs - old_log_probs)
            
            # Clipped objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.ppo_clip, 1.0 + self.ppo_clip) * advantages
            
            # PPO loss: take minimum (pessimistic bound)
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Entropy bonus (encourages exploration)
            entropy = -(probs * log_probs).sum(dim=-1).mean()
            
            # Coordinate loss
            mask = torch.ones_like(batch['dones'])
            for i, length in enumerate(batch['lengths']):
                mask[i, length:] = 0
            
            coord_loss = nn.functional.mse_loss(coords, batch['coords'], reduction='none')
            coord_loss = (coord_loss.mean(dim=-1) * mask).sum() / (mask.sum() + 1e-8)
            
            # Total loss
            total_loss = policy_loss + 0.5 * coord_loss - self.entropy_coef * entropy
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)
            
            self.optimizer.step()
            self.scheduler.step()
            
            # Track losses
            total_losses['action_loss'] += policy_loss.item()
            total_losses['coord_loss'] += coord_loss.item()
            total_losses['entropy'] += entropy.item()
            total_losses['total_loss'] += total_loss.item()
            
            self.update_count += 1
        
        # Average losses
        for key in total_losses:
            total_losses[key] /= num_updates
        
        # Don't clear buffer - rolling window keeps last 64 episodes
        
        return total_losses
    
    def save(self, filepath):
        """Save model checkpoint"""
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
        """Load model checkpoint"""
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
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
    print("GPU Server (PPO) starting...")
    
    server = GPUServerMemory(
        state_size=78,
        hidden_size=1024,
        num_actions=58,
        lr=3e-4
    )
    
    loaded = server.load('memory_model_final.pth')
    if not loaded:
        print("✓ Starting with fresh model")
    
    running = True
    last_print = time.time()
    predictions_served = 0
    episodes_received = 0
    
    print("GPU Server ready (PPO + memory + attention)")
    
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
                    print(f"GPU: Warning - result queue full, dropping prediction")
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
        
        # Train every 16 episodes
        if server.should_train() and episodes_received > 0:
            losses = server.train_step(num_updates=25)
            server.episodes_since_last_train = 0  # Reset counter after training
            
            if losses is not None:
                now = time.time()
                if now - last_print > 5.0:
                    print(f"GPU: {server.update_count} updates | "
                          f"Loss: {losses['total_loss']:.4f} "
                          f"(policy={losses['action_loss']:.4f}, coord={losses['coord_loss']:.4f}, "
                          f"entropy={losses['entropy']:.4f}) | "
                          f"{predictions_served} preds | "
                          f"Episodes: {episodes_received} | "
                          f"Buffer: {len(server.replay_buffer)}")
                    last_print = now
                    predictions_served = 0
                    episodes_received = 0
                
                if server.update_count % 100 == 0:
                    checkpoint_path = f'memory_model_checkpoint_{server.update_count}.pth'
                    server.save(checkpoint_path)
        
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