from gym.envs.registration import register

register(
    id='racecar-v0',
    entry_point='robo_gym.envs:RaceCarEnv',
)
