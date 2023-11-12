from stable_baselines3.common.env_checker import check_env
from light_opt_env_edited import LightOptEnv
from argparse import ArgumentParser

env = LightOptEnv()
# It will check your custom environment and output additional warnings if needed
check_env(env)
