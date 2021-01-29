from gym.envs.registration import register

register(
    id="lift-v0",
    entry_point="gym_linkage.envs:LinkageEnv",
)
