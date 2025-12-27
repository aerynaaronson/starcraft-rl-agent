"""
Pure Win/Loss Reward System

Simple +1 for win, -1 for loss, 0 for timeout.
"""


class RewardCalculator:
    def __init__(self):
        pass
    
    def calculate_step_reward(self, perception, action_name, obs):
        return 0.0
    
    def calculate_terminal_reward(self, outcome, game_length):
        if outcome == 'win':
            return 1.0
        elif outcome == 'loss':
            return 0.0
        else:
            return 0.0
    
    def reset(self):
        pass
    
    def get_base_status(self):
        return "Pure win/loss rewards"


def compute_discounted_rewards(step_rewards, final_reward, gamma=0.99):
    n = len(step_rewards)
    if n == 0:
        return [final_reward]
    
    return [final_reward] * n