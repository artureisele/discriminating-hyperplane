"""
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
"""
import timeit

setup_code = """
import numpy as np
a_h = np.random.rand(1000)
b_h = np.random.rand(1)[0]
r = 0
info = {"bonus": 0}
"""

test_code = """
test_numbers = np.random.uniform(-1, 1, 1000)
percentage_not_filtered = np.mean(a_h * test_numbers < b_h)
r += percentage_not_filtered * 0.5
info["bonus"] += percentage_not_filtered * 0.5
"""

# Measure runtime
print(timeit.timeit(test_code, setup=setup_code, number=1))