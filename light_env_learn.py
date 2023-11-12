from stable_baselines3 import PPO
import os
from light_opt_env import LightOptEnv
import time


import numpy as np
from stable_baselines3 import PPO, TD3
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList


class TensorboardCallback(BaseCallback):
	"""
	Custom callback for plotting additional values in tensorboard.
	"""
	def __init__(self, verbose=0):
		super().__init__(verbose)

	def _on_step(self) -> bool:
		# Log scalar value (here a random variable)
		#print(self.locals)
		
		print("Car_Acc_at_10m", self.locals["new_obs"]["car_acc"][0][0])
		print("Car_Acc_at_20m", self.locals["new_obs"]["car_acc"][0][1])

		print("Ped_Acc_at_10m", self.locals["new_obs"]["ped_acc"][0][0])
		print("Ped_Acc_at_20m", self.locals["new_obs"]["ped_acc"][0][1])

		print("Total_Reward", self.locals["rewards"][0])

		self.logger.record("Car_Acc_at_10m", self.locals["new_obs"]["car_acc"][0][0])
		self.logger.record("Car_Acc_at_20m", self.locals["new_obs"]["car_acc"][0][1])

		self.logger.record("Ped_Acc_at_10m", self.locals["new_obs"]["ped_acc"][0][0])
		self.logger.record("Ped_Acc_at_20m", self.locals["new_obs"]["ped_acc"][0][1])

		self.logger.record("Total_Reward", self.locals["rewards"][0])

		self.logger.dump(step=self.num_timesteps)
		return True


#checkpoint_callback = CheckpointCallback(save_freq=100, save_path="./checkpoints/100_steps_new_reward/")

checkpoint_callback = CheckpointCallback(save_freq=100, save_path="./checkpoints/After_TPE_CMA/1_Trial")

callback = CallbackList([TensorboardCallback(), checkpoint_callback])

env = LightOptEnv()
env.reset()

#model = PPO("MultiInputPolicy", env, tensorboard_log="/PPO_logs/PPO/", verbose=1)
model = TD3("MultiInputPolicy", env, tensorboard_log="C:/Users/grabe/Documents/Rayen/light_distribution_optimization/TD3_logs/1_Trial/", verbose=1)

model.learn(total_timesteps=10000, callback=callback)
