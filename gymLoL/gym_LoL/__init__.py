from gym.envs.registration import register

register(
    id='gymLoL-v0',
    entry_point='gym_LoL.envs:LoLEnv',
)