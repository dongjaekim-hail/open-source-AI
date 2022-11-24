import gym

env = gym.make("LunarLander-v2", render_mode="human")
observation, info = env.reset(seed=42)
for _ in range(1000):
   env.render()
   action = env.action_space.sample()  # this is where you would insert your policy
   observation, reward, terminated, truncated, info = env.step(action)
   print(observation, reward, terminated, truncated, info)

   if terminated or truncated:
      observation, info = env.reset()
env.close()