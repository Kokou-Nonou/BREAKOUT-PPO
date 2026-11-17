import gymnasium as gym
import ale_py

from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation

gym.register_envs(ale_py)


def make_env(render=False):
    if render:
        env = gym.make("ALE/Breakout-v5", render_mode="human", frameskip=1)
    else:
        env = gym.make("ALE/Breakout-v5", frameskip=1)

    env = AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=4,
        screen_size=84,
        grayscale_obs=True,
        grayscale_newaxis=False,
        scale_obs=True,
    )

    env = FrameStackObservation(env, stack_size=4)
    return env