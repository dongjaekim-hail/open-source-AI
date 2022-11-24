#pip install gym
#pip install gym[atari]
#pip install gym[accept-rom-license]
from time import sleep
import gym
env = gym.make("Pong-v4", render_mode='human')
observation, _ = env.reset()
print(observation[0].shape)
print(env.action_space)
print(env.observation_space)

for _ in range(1000):
   action = env.action_space.sample()  # this is where you would insert your policy
   observation, reward, terminated, truncated, info = env.step(action)
   env.render()
   sleep(0.1)
   print(observation, reward, terminated, truncated, info)

   if terminated or truncated:
      observation, info = env.reset()


