import gymnasium as gym
import wandb
wandb.init("test")

env = gym.make("CartPole-v1", render_mode = "rgb_array")
env = gym.wrappers.RecordVideo(env, "videos")
env.reset()
for _ in range(200):
    action = env.action_space.sample()
    env.step(action)

env.close()