import gymnasium as gym
from gymnasium import spaces
import numpy as np
import yfinance as yf
from stable_baselines3 import PPO
from brain_ul import MarketRadar  # folder links the UL model

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

class HybridEnv(gym.Env):
    def __init__(self, processed_data):
        super().__init__()
        self.df = processed_data
        self.current_step = 0
        
        # Action: 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.Discrete(3)
        
        # Observation: [Price, Volume, Signal, Score]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,))

    def _get_obs(self):
        return self.df.iloc[self.current_step].values.astype(np.float32)

    def step(self, action):
        curr_price = self.df.iloc[self.current_step]['Close']
        next_price = self.df.iloc[self.current_step + 1]['Close']
        
        # Reward: Profit/Loss percentage
        reward = (next_price - curr_price) / curr_price if action == 1 else 0
        if action == 2: reward = (curr_price - next_price) / curr_price
        
        self.current_step += 1
        done = self.current_step >= len(self.df) - 2
        return self._get_obs(), reward, done, False, {}

    def reset(self, seed=None):
        self.current_step = 0
        return self._get_obs(), {}

# --- THE LINKED EXECUTION ---
# 1. Get Data
raw_data = yf.download('BTC-INR', period='1d', interval='1m')[['Close', 'Volume']]

# 2. Link with UL (Process the data through the Radar)
radar = MarketRadar()
linked_data = radar.extract_signals(raw_data)

# 3. Link with RL (Feed the UL-enhanced data into the Environment)
env = HybridEnv(linked_data)
agent = PPO("MlpPolicy", env, verbose=1)

print("Starting Hybrid Training...")
agent.learn(total_timesteps=2000)