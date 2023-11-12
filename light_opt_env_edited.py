import gym
import numpy as np
from gym import spaces

from argparse import ArgumentParser
import pandas as pd
import cv2

from unreal_utils import run_unreal_executable
from detection_utils import run_detection


def parse_args():
    # create the parser
    parser = ArgumentParser()
    # add the arguments
    parser.add_argument(
        "--csv_file",
        default=r"C:/Users/grabe/Documents/Rayen/light_distribution_optimization/light/light_csv/Beamer_Optoma_links_2013v3.csv",
        help="path csv file",
    )
    parser.add_argument(
        "--png_file",
        default=r"C:/Users/grabe/Documents/Rayen/light_distribution_optimization/light/light_png/",
        help="path png file",
    )
    parser.add_argument(
        "--exe_dir",
        default=r"C:/Users/grabe/Documents/Rayen/light_distribution_optimization/unreal_engine/UE_Light_Sim_Exe/Windows/UE_Light_Sim.exe",
        help="path to unreal simulation executable",
    )
    # parse the arguments
    args = parser.parse_args()
    return args


class LightOptEnv(gym.Env):
    def __init__(self):
        super().__init__()

        # Defince action space
        self.action_space = spaces.Box(0, 1, (64,), dtype=np.float32)

        # Define observation_space
        self.observation_space = spaces.Dict(
            spaces={
                "light_dist": spaces.Box(0, 1, (64,), dtype=np.float32),
                "ped_acc": spaces.Box(0, 1, shape=(2,), dtype=np.float32),
                "car_acc": spaces.Box(0, 1, shape=(2,), dtype=np.float32),
            }
        )

        # Randomize the current and next state
        self.current_state = self.observation_space.sample()
        self.next_state = self.observation_space.sample()

        # Define positions list
        self.pos_list = [10, 20]

        # Initialze steps
        self.count = 0
        self.max_count = 100
        self.done = False

        # Get the args
        self.args = parse_args()

    def reset(self):
        # Initialze steps
        self.count = 0
        self.max_count = 100
        self.done = False

        self.current_state["light_dist"] = np.zeros((64,), dtype=np.float32)
        self.current_state["ped_acc"] = np.zeros((2,), dtype=np.float32)
        self.current_state["car_acc"] = np.zeros((2,), dtype=np.float32)
        
        return self.current_state  # reward, done, info can't be included

    def step(self, action):
        print(action)
        # Convert the action to png
        cv2.imwrite(
            self.args.png_file + "/Light_distribution_rechts.png",
            action.reshape(8, 8) * 255,
        )
        cv2.imwrite(
            self.args.png_file + "/Light_distribution_links.png",
            action.reshape(8, 8) * 255,
        )
        # Run the unreal engine executable
        run_unreal_executable(self.args.exe_dir)
        # Run detection on the generated images
        run_detection()

        # Get the next state
        self.next_state["light_dist"] = action
        # Get accuracy
        next_ped_acc, next_car_acc = self.get_accuracy()
        self.next_state["ped_acc"] = next_ped_acc
        self.next_state["car_acc"] = next_car_acc

        # Update step
        self.count += 1

        # Calculate reward
        reward, self.done = self.calc_reward(action)

        # Assign next_state as current_state
        self.current_state["light_dist"] = self.next_state["light_dist"]
        self.current_state["ped_acc"] = self.next_state["ped_acc"]
        self.current_state["car_acc"] = self.next_state["car_acc"]

        
        return self.current_state, reward, self.done, {}

    def get_accuracy(self):
        df_result = pd.read_csv(
            r"C:/Users/grabe/Documents/Rayen/light_distribution_optimization/object_detection/Object_Detection_Results.csv"
        )

        pedestrian_confidence = (
            df_result[df_result["class"] == "pedestrian"]
            .groupby("distance")["confidence_score"]
            .mean()
        )

        car_confidence = (
            df_result[df_result["class"] == "car"]
            .groupby("distance")["confidence_score"]
            .mean()
        )

        ped_arr = np.array(pedestrian_confidence.values, dtype=np.float32)
        car_arr = np.array(car_confidence.values, dtype=np.float32)

        return ped_arr, car_arr

    def calc_reward(self, action):
        # Weights for ped and car
        W_ped = 1
        W_car = 0.8
        # Weights for positions
        W_pos = [1, 0.8]
        # Declare reward and total_reward
        reward = 0
        #part_reward = 0

        for i in range(len(self.pos_list)):
            part_reward = 0
            # Extract the accuracy values for current and next positions
            e_p_current = 1 - self.current_state["ped_acc"][i]
            e_c_current = 1 - self.current_state["car_acc"][i]
            e_p_next = 1 - self.next_state["ped_acc"][i]
            e_c_next = 1 - self.next_state["car_acc"][i]

            # Calculate the delta for each class
            delta_e_p = e_p_next - e_p_current
            delta_e_c = e_c_next - e_c_current

            print("Position " + str(self.pos_list[i]) + ": \n")
            print("Pedestrian: ")
            print("Current state: ", self.current_state["ped_acc"][i])
            print("Next state: ", self.next_state["ped_acc"][i])
            print("e_p_next: ", e_p_next)

            print("Car: ")
            print("Current state: ", self.current_state["car_acc"][i])
            print("Next state: ", self.next_state["car_acc"][i])
            print("e_c_next: ", e_c_next)

            print("delta_e_p: ", delta_e_p)
            print("delta_e_c: ", delta_e_c)

            # Calculate the reward
            if delta_e_p < 0 and delta_e_c < 0:
                part_reward = (W_ped * W_pos[i] * delta_e_p) ** 2 + (
                    W_car * W_pos[i] * delta_e_c
                ) ** 2
            elif delta_e_p >= 0 and delta_e_c >= 0:
                part_reward = (
                    -((W_ped * W_pos[i] * delta_e_p) ** 2)
                    - (W_car * W_pos[i] * delta_e_c) ** 2
                    - 0.1
                )
            
            # if e_p_next <= 0.2 and e_c_next <=0.2:
            #     part_reward += 5
            
            print("Part_Reward " + str(self.pos_list[i]) + ": ")
            print(part_reward)

            reward += part_reward

        print("Reward before self_count/action/9:", reward)

        if self.count > self.max_count:
            reward = -10
            self.done=True
        
        reward -= action.mean()

        if reward>2:    
            self.done=True
        
        print("Reward after self_count/action/9:", reward)
                        
        return reward, self.done
