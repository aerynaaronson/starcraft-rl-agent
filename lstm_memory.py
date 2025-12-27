"""
DROP-IN LSTM OPTIMIZATION FOR SINGLE WORKER

RTX 3090 (24GB VRAM) + Ryzen 5900x + 64GB RAM
Step limit: 10,000

UPDATED: 58 actions (was 61)
- Removed: attack_move, all unit transformations
- Added: 6 strategic combat macros
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque

ACTION_TO_INDEX = {
    # Basic
    "do_nothing": 0,
    
    # Worker Management
    "send_idle_workers_to_mine": 1,
    "send_idle_workers_to_gas": 2,
    "train_worker": 3,
    
    # Core Buildings
    "build_supply_depot": 4,
    "build_command_center": 5,
    "build_barracks": 6,
    "build_factory": 7,
    "build_starport": 8,
    "build_refinery": 9,
    
    # Tech Buildings
    "build_engineering_bay": 10,
    "build_armory": 11,
    "build_fusion_core": 12,
    "build_ghost_academy": 13,
    
    # Defensive Buildings
    "build_bunker": 14,
    "build_missile_turret": 15,
    "build_sensor_tower": 16,
    
    # Addons
    "add_barracks_techlab": 17,
    "add_barracks_reactor": 18,
    "add_factory_techlab": 19,
    "add_factory_reactor": 20,
    "add_starport_techlab": 21,
    "add_starport_reactor": 22,
    "build_techlab": 23,
    "build_reactor": 24,
    
    # Building Upgrades
    "upgrade_orbital_command": 25,
    "upgrade_planetary_fortress": 26,
    "lower_depot": 27,
    
    # Barracks Units
    "train_marine": 28,
    "train_marauder": 29,
    "train_reaper": 30,
    "train_ghost": 31,
    
    # Factory Units
    "train_hellion": 32,
    "train_hellbat": 33,
    "train_siege_tank": 34,
    "train_thor": 35,
    "train_cyclone": 36,
    "train_widow_mine": 37,
    
    # Starport Units
    "train_viking": 38,
    "train_medivac": 39,
    "train_liberator": 40,
    "train_raven": 41,
    "train_banshee": 42,
    "train_battlecruiser": 43,
    
    # Abilities
    "call_down_mule": 44,
    "scanner_sweep": 45,
    
    # Research
    "research_barracks_techlab": 46,
    "research_factory_techlab": 47,
    "research_starport_techlab": 48,
    "research_engineering_bay": 49,
    "research_armory": 50,
    "research_ghost_academy": 51,
    
    # Combat - STRATEGIC MACROS (was 1 action, now 6)
    "attack_aggressor": 52,
    "attack_siege": 53,
    "attack_support": 54,
    "attack_harass": 55,
    "retreat_to_base": 56,
    "final_push": 57,
}

class PositionalEncoding(nn.Module):
    """Positional encoding for temporal information"""
    def __init__(self, d_model, max_len=15000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x, step_idx=0):
        """
        Args:
            x: [batch, seq_len, d_model] or [batch, d_model]
            step_idx: current step index for single-step inference
        """
        if len(x.shape) == 3:
            return x + self.pe[:x.size(1), :].unsqueeze(0)
        else:
            # Clip step_idx to buffer size
            step_idx = min(step_idx, self.pe.size(0) - 1)
            return x + self.pe[step_idx, :]


class TemporalAttention(nn.Module):
    """Multi-head attention to attend to important past events"""
    def __init__(self, hidden_size, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads
        
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        
        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, hidden_size)
        self.out_linear = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, query, key_value_memory):
        """
        Args:
            query: [batch, hidden_size] - current state
            key_value_memory: [batch, seq_len, hidden_size] - past states
        """
        batch_size = query.size(0)
        seq_len = key_value_memory.size(1)
        
        # Project and reshape for multi-head attention
        Q = self.q_linear(query).view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(key_value_memory).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(key_value_memory).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, 1, self.hidden_size)
        
        # Output projection
        output = self.out_linear(context.squeeze(1))
        
        return output, attn_weights


class EpisodeMemoryBuffer:
    """Buffer to store episode history for attention mechanism"""
    def __init__(self, max_length=2000):
        self.max_length = max_length
        self.states = deque(maxlen=max_length)
        self.lstm_outputs = deque(maxlen=max_length)
        
    def add(self, state, lstm_output):
        """Add a step to memory"""
        self.states.append(state)
        self.lstm_outputs.append(lstm_output)
    
    def get_memory_tensor(self, device):
        """Get memory as tensor for attention"""
        if len(self.lstm_outputs) == 0:
            return None
        memory = torch.stack(list(self.lstm_outputs)).to(device)
        return memory.unsqueeze(0)  # [1, seq_len, hidden_size]
    
    def clear(self):
        """Clear memory at episode end"""
        self.states.clear()
        self.lstm_outputs.clear()
    
    def __len__(self):
        return len(self.lstm_outputs)


class LSTMActionNetwork(nn.Module):
    """
    LSTM network with temporal attention for long-term memory.
    
    OPTIMIZED FOR SINGLE WORKER (RTX 3090, 10k step limit):
    - Memory buffer: 10000 steps (full episode memory)
    - Hidden size: 1024 (more capacity, no worker split)
    - LSTM layers: 4 (deeper understanding)
    - Actions: 58 (was 61 - removed transformations, added 6 combat macros)
    - Parameters: ~50M (fits easily in 24GB VRAM)
    """
    
    def __init__(self, state_size=78, hidden_size=1024, num_actions=58, 
                 num_lstm_layers=4, num_attention_heads=8, use_attention=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_actions = num_actions
        self.num_lstm_layers = num_lstm_layers
        self.use_attention = use_attention
        
        # Input projection to match hidden size
        self.input_proj = nn.Linear(state_size, hidden_size)
        
        # Positional encoding for temporal information
        self.pos_encoding = PositionalEncoding(hidden_size, max_len=10000)
        
        # Multi-layer LSTM for sequential processing
        self.lstm = nn.LSTM(
            hidden_size, 
            hidden_size, 
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=0.1 if num_lstm_layers > 1 else 0
        )
        
        # Layer normalization for stable training
        self.ln1 = nn.LayerNorm(hidden_size)
        
        # Temporal attention to look back at episode history
        if use_attention:
            self.attention = TemporalAttention(hidden_size, num_heads=num_attention_heads)
            self.ln2 = nn.LayerNorm(hidden_size)
        
        # Action head with larger capacity
        self.fc_action = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions)
        )
        
        # Coordinate head
        self.coord_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 2),
            nn.Sigmoid()  # [0, 1] range
        )
        
        # Episode memory buffer for attention - FULL EPISODE
        self.episode_memory = EpisodeMemoryBuffer(max_length=10000)
        self.current_step = 0

    def init_hidden(self, batch_size=1):
        """Initialize hidden state for LSTM"""
        device = next(self.parameters()).device
        h0 = torch.zeros(self.num_lstm_layers, batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(self.num_lstm_layers, batch_size, self.hidden_size, device=device)
        return (h0, c0)
    
    def reset_episode(self):
        """Reset episode memory at the start of a new episode"""
        self.episode_memory.clear()
        self.current_step = 0
    
    def forward(self, state, hidden=None, use_memory=True):
        """
        Forward pass with temporal attention.
        
        Args:
            state: State tensor [batch, state_size] or [batch, seq_len, state_size]
            hidden: Optional LSTM hidden state
            use_memory: Whether to use attention over episode memory
            
        Returns:
            action_logits: Action logits [batch, num_actions]
            coords: Normalized coordinates [batch, 2] in [0, 1] range
            hidden: Updated LSTM hidden state
            attention_weights: Attention weights over history (if use_attention=True)
        """
        device = state.device
        
        # Handle single state vs sequence
        is_single_state = len(state.shape) == 2
        if is_single_state:
            state = state.unsqueeze(1)  # [batch, 1, state_size]
        
        batch_size, seq_len, _ = state.shape
        
        # Initialize hidden state if not provided
        if hidden is None:
            hidden = self.init_hidden(batch_size)
        
        # Project input to hidden size
        x = self.input_proj(state)  # [batch, seq_len, hidden_size]
        
        # Add positional encoding
        if is_single_state:
            x = self.pos_encoding(x.squeeze(1), step_idx=self.current_step).unsqueeze(1)
        else:
            x = self.pos_encoding(x)
        
        # LSTM forward pass
        lstm_out, hidden = self.lstm(x, hidden)  # [batch, seq_len, hidden_size]
        lstm_out = self.ln1(lstm_out)
        
        # Get final output for this step
        current_output = lstm_out[:, -1, :]  # [batch, hidden_size]
        
        # Store in episode memory for attention
        if is_single_state and use_memory:
            self.episode_memory.add(state[:, 0, :], current_output.detach())
            self.current_step += 1
        
        # Temporal attention over episode history
        attention_weights = None
        if self.use_attention and use_memory and len(self.episode_memory) > 1:
            memory_tensor = self.episode_memory.get_memory_tensor(device)
            if memory_tensor is not None:
                attn_output, attention_weights = self.attention(current_output, memory_tensor)
                current_output = current_output + attn_output  # Residual connection
                current_output = self.ln2(current_output)
        
        # Action and coordinate heads
        action_logits = self.fc_action(current_output)
        coords = self.coord_head(current_output)  # Already [0, 1] from sigmoid
        
        return action_logits, coords, hidden, attention_weights
    
    def predict(self, state, legal_actions, action_to_index, coord_transform, hidden=None):
        """
        Predict action and coordinates for a given state.
        
        Args:
            state: State array
            legal_actions: List of legal action names
            action_to_index: Action name to index mapping
            coord_transform: CoordinateTransform instance
            hidden: Optional LSTM hidden state
            
        Returns:
            action_name: Name of selected action
            x_norm: Normalized x coordinate [0, 1]
            y_norm: Normalized y coordinate [0, 1]
            x_world: World x coordinate (integer)
            y_world: World y coordinate (integer)
            hidden: Updated LSTM hidden state
        """
        self.eval()
        with torch.no_grad():
            device = next(self.parameters()).device
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            
            if hidden is not None:
                hidden = (hidden[0].to(device), hidden[1].to(device))
            
            # Forward pass
            action_logits, coords, hidden, attn_weights = self(state_tensor, hidden, use_memory=True)
            
            # Mask illegal actions
            action_logits = action_logits.squeeze(0)
            mask = torch.full((self.num_actions,), float('-inf'), device=device)
            legal_indices = [action_to_index[a] for a in legal_actions if a in action_to_index]
            mask[legal_indices] = 0
            action_logits = action_logits + mask
            action_idx = torch.argmax(action_logits).item()
            
            # Extract coordinates - already in [0, 1]
            x_norm, y_norm = coords.squeeze(0).cpu().numpy()
            x_norm = float(np.clip(x_norm, 0, 1))
            y_norm = float(np.clip(y_norm, 0, 1))
            
            # Convert to world coordinates
            x_world, y_world = coord_transform.normalized_to_world(x_norm, y_norm)
            
            # Get action name
            index_to_action = {v: k for k, v in action_to_index.items()}
            action_name = index_to_action.get(action_idx, "do_nothing")
            
            return action_name, x_norm, y_norm, x_world, y_world, hidden


class SequenceReplayBuffer:
    """
    Replay buffer that stores sequences for training.
    Supports full episode sequences and truncated BPTT.
    """
    def __init__(self, max_episodes=1000):
        self.episodes = deque(maxlen=max_episodes)
    
    def add_episode(self, states, actions, coords, rewards, dones):
        """
        Add a complete episode to the buffer.
        
        Args:
            states: List of state arrays
            actions: List of action indices
            coords: List of (x, y) coordinate tuples
            rewards: List of rewards
            dones: List of done flags
        """
        episode = {
            'states': np.array(states),
            'actions': np.array(actions),
            'coords': np.array(coords),
            'rewards': np.array(rewards),
            'dones': np.array(dones)
        }
        self.episodes.append(episode)
    
    def clear(self):
        self.episodes.clear()
    
    def sample_sequences(self, batch_size, sequence_length, device='cpu'):
        """
        Sample random sequences from episodes.
        
        Args:
            batch_size: Number of sequences to sample
            sequence_length: Length of each sequence
            device: Device to put tensors on
            
        Returns:
            Dictionary of batched tensors
        """
        if len(self.episodes) == 0:
            return None
        
        sequences = []
        for _ in range(batch_size):
            # Sample random episode
            episode = self.episodes[np.random.randint(len(self.episodes))]
            ep_len = len(episode['states'])
            
            if ep_len <= sequence_length:
                # Use full episode if shorter than sequence_length
                start_idx = 0
                seq_len = ep_len
            else:
                # Sample random subsequence
                start_idx = np.random.randint(0, ep_len - sequence_length)
                seq_len = sequence_length
            
            sequences.append({
                'states': episode['states'][start_idx:start_idx + seq_len],
                'actions': episode['actions'][start_idx:start_idx + seq_len],
                'coords': episode['coords'][start_idx:start_idx + seq_len],
                'rewards': episode['rewards'][start_idx:start_idx + seq_len],
                'dones': episode['dones'][start_idx:start_idx + seq_len]
            })
        
        # Pad sequences to same length and convert to tensors
        max_len = max(len(seq['states']) for seq in sequences)
        
        batch = {
            'states': [],
            'actions': [],
            'coords': [],
            'rewards': [],
            'dones': [],
            'lengths': []
        }
        
        for seq in sequences:
            seq_len = len(seq['states'])
            pad_len = max_len - seq_len
            
            batch['states'].append(np.pad(seq['states'], ((0, pad_len), (0, 0)), mode='constant'))
            batch['actions'].append(np.pad(seq['actions'], (0, pad_len), mode='constant'))
            batch['coords'].append(np.pad(seq['coords'], ((0, pad_len), (0, 0)), mode='constant'))
            batch['rewards'].append(np.pad(seq['rewards'], (0, pad_len), mode='constant'))
            batch['dones'].append(np.pad(seq['dones'], (0, pad_len), mode='constant'))
            batch['lengths'].append(seq_len)
        
        # Convert to tensors
        return {
            'states': torch.tensor(np.array(batch['states']), dtype=torch.float32, device=device),
            'actions': torch.tensor(np.array(batch['actions']), dtype=torch.long, device=device),
            'coords': torch.tensor(np.array(batch['coords']), dtype=torch.float32, device=device),
            'rewards': torch.tensor(np.array(batch['rewards']), dtype=torch.float32, device=device),
            'dones': torch.tensor(np.array(batch['dones']), dtype=torch.float32, device=device),
            'lengths': batch['lengths']
        }
    
    def __len__(self):
        return len(self.episodes)


def get_spawn_corner(spawn_x, spawn_y, map_width, map_height):
    """Determine which corner the spawn is in."""
    is_left = spawn_x < map_width / 2
    is_top = spawn_y < map_height / 2
    if is_top and is_left:
        return 'top_left'
    elif is_top and not is_left:
        return 'top_right'
    elif not is_top and is_left:
        return 'bottom_left'
    else:
        return 'bottom_right'