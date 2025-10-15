import gymnasium as gym
from rocket_env import SimpleRocketEnv
from stable_baselines3 import DQN


if __name__ == "__main__":
    env = SimpleRocketEnv(render_mode=None)

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=64,
        tau=0.01,
        gamma=0.99,
        train_freq=4,
        target_update_interval=1000,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        exploration_fraction=0.3,
        verbose=1,
    )

    # Latih agent
    model.learn(total_timesteps=100000)

    # Simpan model
    model.save("dqn_rocket")
    env.close()
