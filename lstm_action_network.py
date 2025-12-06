import numpy as np
import pickle

ACTION_TO_INDEX = {
    "do_nothing": 0,
    "send_idle_workers_to_mine": 1,
    "train_worker": 2,
    "build_supply_depot": 3,
    "build_barracks": 4,
    "train_marine": 5,
    "attack_move": 6,
    "lower_depot": 7,
    "build_command_center": 8,
    "build_refinery": 9,
    "send_idle_workers_to_gas": 10,
    "build_techlab": 11,
    "build_reactor": 12,
    "train_marauder": 13,
    "train_reaper": 14,
    "build_factory": 15,
    "train_hellion": 16,
    "add_factory_techlab": 17,
    "add_factory_reactor": 18,
    "build_armory": 19,
    "train_siege_tank": 20,
    "train_thor": 21,
    "train_cyclone": 22,
    "build_starport": 23,
    "build_fusion_core": 24,
    "add_starport_techlab": 25,
    "add_starport_reactor": 26,
    "train_viking": 27,
    "train_medivac": 28,
    "train_liberator": 29,
    "train_raven": 30,
    "train_banshee": 31,
    "train_battlecruiser": 32,
    "add_barracks_techlab": 33,
    "add_barracks_reactor": 34,
}

class LSTMActionNetwork:
    """
    LSTM-based network for StarCraft II action selection.
    Maintains hidden state across timesteps within an episode.
    """
    def __init__(self, state_size=35, hidden_size=128, num_actions=35, learning_rate=0.01):
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        
        # LSTM parameters (simplified single-layer LSTM)
        # Input gate
        self.W_i = np.random.randn(state_size, hidden_size) * 0.1
        self.U_i = np.random.randn(hidden_size, hidden_size) * 0.1
        self.b_i = np.zeros(hidden_size)
        
        # Forget gate
        self.W_f = np.random.randn(state_size, hidden_size) * 0.1
        self.U_f = np.random.randn(hidden_size, hidden_size) * 0.1
        self.b_f = np.zeros(hidden_size)
        
        # Output gate
        self.W_o = np.random.randn(state_size, hidden_size) * 0.1
        self.U_o = np.random.randn(hidden_size, hidden_size) * 0.1
        self.b_o = np.zeros(hidden_size)
        
        # Cell state
        self.W_c = np.random.randn(state_size, hidden_size) * 0.1
        self.U_c = np.random.randn(hidden_size, hidden_size) * 0.1
        self.b_c = np.zeros(hidden_size)
        
        # Output layer (hidden -> actions)
        self.W_out = np.random.randn(hidden_size, num_actions) * 0.1
        self.b_out = np.zeros(num_actions)
        
        # Coordinate prediction (per action)
        self.coord_weights = np.random.randn(num_actions, 2) * 0.1
        
        # Hidden state (reset at episode start)
        self.h = np.zeros(hidden_size)
        self.c = np.zeros(hidden_size)
    
    def reset_hidden_state(self):
        """Call this at the start of each episode"""
        self.h = np.zeros(self.hidden_size)
        self.c = np.zeros(self.hidden_size)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def tanh(self, x):
        return np.tanh(np.clip(x, -500, 500))
    
    def lstm_step(self, x):
        """
        Single LSTM forward step
        x: input state vector
        Returns: new hidden state
        """
        # Input gate
        i_t = self.sigmoid(x @ self.W_i + self.h @ self.U_i + self.b_i)
        
        # Forget gate
        f_t = self.sigmoid(x @ self.W_f + self.h @ self.U_f + self.b_f)
        
        # Output gate
        o_t = self.sigmoid(x @ self.W_o + self.h @ self.U_o + self.b_o)
        
        # Cell state candidate
        c_tilde = self.tanh(x @ self.W_c + self.h @ self.U_c + self.b_c)
        
        # Update cell state
        self.c = f_t * self.c + i_t * c_tilde
        
        # Update hidden state
        self.h = o_t * self.tanh(self.c)
        
        return self.h
    
    def encode_state(self, perception):
        return np.array([
            perception.minerals / 1000.0,
            perception.vespene / 1000.0,
            perception.supply_used / 200.0,
            perception.supply_cap / 200.0,
            perception.scv_count() / 50.0,
            perception.marine_count() / 50.0,
            perception.depot_count() / 10.0,
            perception.barracks_count() / 10.0,
            perception.cc_count() / 3.0,
            1.0 if perception.supply_available() else 0.0,
            1.0 if perception.supply_full() else 0.0,
            perception.techlab_count() / 5.0,
            perception.reactor_count() / 5.0,
            perception.marauder_count() / 50.0,
            perception.reaper_count() / 50.0,
            perception.factory_count() / 5.0,
            perception.hellion_count() / 50.0,
            perception.factory_techlab_count() / 5.0,
            perception.factory_reactor_count() / 5.0,
            perception.armory_count() / 5.0,
            perception.siege_tank_count() / 50.0,
            perception.thor_count() / 50.0,
            perception.cyclone_count() / 50.0,
            perception.starport_count() / 5.0,
            perception.fusion_core_count() / 3.0,
            perception.starport_techlab_count() / 5.0,
            perception.starport_reactor_count() / 5.0,
            perception.viking_count() / 50.0,
            perception.medivac_count() / 50.0,
            perception.liberator_count() / 50.0,
            perception.raven_count() / 50.0,
            perception.banshee_count() / 50.0,
            perception.battlecruiser_count() / 20.0,
            perception.barracks_techlab_count() / 5.0,
            perception.barracks_reactor_count() / 5.0,
        ])
    
    def predict(self, state, legal_actions, action_to_index):
        # LSTM forward pass
        h = self.lstm_step(state)
        
        # Action values from hidden state
        action_values = h @ self.W_out + self.b_out
        
        # Mask illegal actions
        legal_mask = np.full(self.num_actions, -1e10)
        legal_indices = [action_to_index[a] for a in legal_actions if a in action_to_index]
        legal_mask[legal_indices] = 0
        action_values += legal_mask
        
        chosen_index = np.argmax(action_values)
        index_to_action = {v: k for k, v in action_to_index.items()}
        action_name = index_to_action.get(chosen_index, "do_nothing")
        
        # Coordinate prediction
        x_norm, y_norm = self.coord_weights[chosen_index]
        x_norm = float(max(0.0, min(1.0, x_norm)))
        y_norm = float(max(0.0, min(1.0, y_norm)))
        
        return action_name, x_norm, y_norm
    
    def update_game_outcome(self, history, outcome, game_time_seconds):
        """
        Simplified BPTT (Backpropagation Through Time)
        For now, using simple gradient updates like the linear version
        Full BPTT would be more complex
        """
        # Pure win/loss (remove time penalty if desired)
        reward = 100 if outcome == 'win' else -100
        
        for state, action_index, x_norm, y_norm in history:
            # Simple gradient update (not true BPTT, but keeps it similar to your original)
            # For full LSTM training, you'd need to store all hidden states
            # and do proper backprop through time
            
            # Update output layer
            self.W_out[:, action_index] += self.learning_rate * reward * self.h
            self.b_out[action_index] += self.learning_rate * reward
            
            # Update coordinate weights
            self.coord_weights[action_index] += self.learning_rate * reward * np.array([x_norm, y_norm])
    
    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump({
                'W_i': self.W_i, 'U_i': self.U_i, 'b_i': self.b_i,
                'W_f': self.W_f, 'U_f': self.U_f, 'b_f': self.b_f,
                'W_o': self.W_o, 'U_o': self.U_o, 'b_o': self.b_o,
                'W_c': self.W_c, 'U_c': self.U_c, 'b_c': self.b_c,
                'W_out': self.W_out, 'b_out': self.b_out,
                'coord_weights': self.coord_weights
            }, f)
    
    def load(self, filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.W_i = data['W_i']
            self.U_i = data['U_i']
            self.b_i = data['b_i']
            self.W_f = data['W_f']
            self.U_f = data['U_f']
            self.b_f = data['b_f']
            self.W_o = data['W_o']
            self.U_o = data['U_o']
            self.b_o = data['b_o']
            self.W_c = data['W_c']
            self.U_c = data['U_c']
            self.b_c = data['b_c']
            self.W_out = data['W_out']
            self.b_out = data['b_out']
            self.coord_weights = data.get('coord_weights', np.random.randn(self.num_actions, 2) * 0.1)