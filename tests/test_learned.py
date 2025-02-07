import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env

import logging
import shutil as sh
from PIL import Image
import datetime
from pytz import timezone
from multiprocessing import freeze_support

TZ = timezone('Europe/Moscow')


def save_mp4(eps_frames, _path_filename, fps=25):
    eps_frame_dir = 'episode_frames'
    os.mkdir(eps_frame_dir)

    for i, frame in enumerate(eps_frames):
        Image.fromarray(frame).save(os.path.join(eps_frame_dir, f'frame-{i + 1}.png'))

    os.system(f'ffmpeg -v 0 -r {fps} -i {eps_frame_dir}/frame-%1d.png -vcodec libx264 -b 10M -y "{_path_filename}"');
    sh.rmtree(eps_frame_dir)


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
    path_filename = os.path.join(main_path, env_id)

    eval_vec_env_kwargs = dict(env_id=env_id,
                               env_kwargs=env_kwargs,
                               n_envs=1,
                               seed=42,
                               vec_env_cls=DummyVecEnv)
    # Load the model
    eval_vec_env = make_vec_env(**eval_vec_env_kwargs)
    model = PPO.load(path=path_filename, env=eval_vec_env, device="cpu")
    env = model.get_env()

    # Test the model
    obs = env.reset()
    frames = []
    while True:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        one_frame = env.render(mode='rgb_array')  # Render the picture as an RGB array
        frames.append(one_frame)
        if done:
            break
    save_mp4(frames, os.path.join(main_path, f"{env_id}_{exp_id}.mp4"))
