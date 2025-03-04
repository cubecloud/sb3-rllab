import os
import logging
import datetime
import numpy as np

from sb3_contrib import RecurrentPPO
# from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import StackedObservations
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env

from pytz import timezone
from multiprocessing import freeze_support
from sb3_rllab import add_score, save_mp4

TZ = timezone('Europe/Moscow')


if __name__ == "__main__":
    freeze_support()

    logger = logging.getLogger(f'test_labsubprocvecenv.log')
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    lookback = 20
    env_id = "LunarLander-v2"
    env_kwargs = dict(render_mode="rgb_array",
                      continuous=True,
                      gravity=-9.8,
                      enable_wind=True,
                      wind_power=15.0,
                      turbulence_power=1.5)

    exp_id = f'{datetime.datetime.now(TZ).strftime("%y%d%m-%H%M%S")}'
    main_path = "."

    eval_vec_env_kwargs = dict(env_id=env_id,
                               env_kwargs=env_kwargs,
                               n_envs=1,
                               seed=42,
                               vec_env_cls=DummyVecEnv)

    # Load the model
    eval_vec_env = make_vec_env(**eval_vec_env_kwargs)
    eval_vec_env = VecFrameStack(eval_vec_env, n_stack=lookback)
    policy_id = f'{RecurrentPPO.__name__}'
    path_filename = os.path.join(main_path, f"{policy_id}_{env_id}")
    model = RecurrentPPO.load(path=os.path.join(path_filename, 'best_model', 'best_model.zip'), env=eval_vec_env,
                              device="cpu")

    env = model.get_env()

    obs = env.reset()
    score = 0
    frames = []
    lstm_states = None
    num_envs = 1
    episode_starts = np.ones((num_envs,), dtype=bool)
    while True:
        action, _states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=False)
        obs, reward, dones, info = env.step(action)
        score += reward
        one_frame = env.render(mode="rgb_array")  # Render the picture as an RGB array
        one_frame = add_score(one_frame, score)
        frames.append(one_frame)
        episode_starts = dones
        if any(dones):
            break
    save_mp4(frames, os.path.join(main_path, f"{policy_id}_{env_id}_{exp_id}.mp4"), fps=22)
