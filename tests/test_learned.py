import os
import logging
import datetime

from stable_baselines3 import PPO
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
    policy_id = f'{PPO.__name__}'
    path_filename = os.path.join(main_path, f"{policy_id}_{env_id}")
    model = PPO.load(path=os.path.join(path_filename, 'best_model', 'best_model.zip'), env=eval_vec_env, device="cpu")

    env = model.get_env()

    # Test the model
    obs = env.reset()
    score = 0
    frames = []
    while True:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        score += reward
        one_frame = env.render(mode="rgb_array")  # Render the picture as an RGB array
        one_frame = add_score(one_frame, score)
        frames.append(one_frame)
        if done:
            break
    save_mp4(frames, os.path.join(main_path, f"{policy_id}_{env_id}_{exp_id}.mp4"), fps=22)
