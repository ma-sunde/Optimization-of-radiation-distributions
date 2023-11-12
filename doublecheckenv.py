from light_opt_env_edited import LightOptEnv


env = LightOptEnv()
episodes = 10

for episode in range(episodes):
    done = False
    obs = env.reset()
    while True:  # not done:
        random_action = env.action_space.sample()
        print("action", random_action)
        obs, reward, done, info = env.step(random_action)
        print("reward", reward)
