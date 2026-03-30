from gymnasium.envs.registration import register
from env.user_sim import UserSimEnv

register(
    id='user_env',
    entry_point=UserSimEnv,
)