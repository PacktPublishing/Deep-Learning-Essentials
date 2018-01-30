import gym
import universe


if __name__ == "__main__":
    env = gym.make('flashgames.DuskDrive-v0')
    env.configure(remotes=1)
    observations = env.reset()
    while True:
        action = [[('KeyEvent', 'ArrowUp', True)] for obs in observations]
        observation, reward, done, info = env.step(action)
        env.render()
