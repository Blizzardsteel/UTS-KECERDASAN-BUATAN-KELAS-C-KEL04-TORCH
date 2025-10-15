import pygame
from stable_baselines3 import DQN
from rocket_env import SimpleRocketEnv

if __name__ == "__main__":
    env = SimpleRocketEnv()
    model = DQN.load("dqn_rocket")  # ← ini pakai model hasil training

    obs, _ = env.reset()
    done = False

    while not done:
        # Model memilih action
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        env.render()

    env.close()
