import os
from light_opt_env import LightOptEnv
import time
import numpy as np
from stable_baselines3 import TD3

from stable_baselines3.common.evaluation import evaluate_policy
import pandas as pd


env = LightOptEnv()

# Load the trained agent
# NOTE: if you have loading issue, you can pass `print_system_info=True`
# to compare the system on which the model was trained vs the current one
# model = DQN.load("dqn_lunar", env=env, print_system_info=True)
model = TD3.load("C:/Users/grabe/Documents/Rayen/light_distribution_optimization/checkpoints/After_TPE_CMA/1_Trial/rl_model_6100_steps.zip", env=env)

# Evaluate the agent
# NOTE: If you use wrappers with your environment that modify rewards,
#       this will be reflected here. To evaluate with original rewards,
#       wrap environment in a "Monitor" wrapper before other wrappers.
#mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

# Initialize a list to store the step indices and total_err_acc
step_list = []
total_err_acc_list = []
next_ped_acc_list = []
next_car_acc_list = []

# Enjoy trained agent
vec_env = model.get_env()
obs = vec_env.reset()
for i in range(50):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)


    total_err_acc = info[0]["total_err_acc"]
    next_ped_acc = info[0]["next_ped_acc"]
    next_car_acc = info[0]["next_car_acc"]
    
    # Append the step index and total_err_acc to the lists
    step_list.append(i)
    total_err_acc_list.append(total_err_acc)
    next_ped_acc_list.append(next_ped_acc)
    next_car_acc_list.append(next_car_acc)

    # Print the step index and total_err_acc for this step
    print(f"Step {i}: total_err_acc={total_err_acc}, next_ped_acc={next_ped_acc}, next_car_acc={next_car_acc}")

df = pd.DataFrame({
    "Step": step_list,
    "Total Error Accuracy": total_err_acc_list,
    "Next Pedestrian Accuracy": next_ped_acc_list,
    "Next Car Accuracy": next_car_acc_list
})

# Write the dataframe to a CSV file
df.to_csv("total_err_acc_load_model.csv", index=False)