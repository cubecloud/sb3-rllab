import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from sb3_rllab import LabSubprocVecEnv
from sb3_rllab import CoScheduler

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

    seed = 443
    num_envs = 300
    total_timesteps = 20_000_000

    env_id = "LunarLander-v2"
    env_kwargs = dict(render_mode=None,  # render off
                      continuous=True,
                      gravity=-9.8,
                      enable_wind=True,
                      wind_power=15.0,
                      turbulence_power=1.5)

    train_vec_env_kwargs = dict(env_id=env_id,
                                env_kwargs=env_kwargs,
                                n_envs=num_envs,
                                seed=seed,
                                vec_env_cls=LabSubprocVecEnv,
                                vec_env_kwargs=dict(use_threads=True,
                                                    use_period='train')
                                )

    train_vec_env = make_vec_env(**train_vec_env_kwargs)

    exp_id = f'{datetime.datetime.now(TZ).strftime("%y%d%m-%H%M%S")}'
    main_path = "."
    path_filename = os.path.join(main_path, env_id)

    n_steps = 200
    model = PPO("MlpPolicy",
                train_vec_env,
                verbose=1,
                device="auto",
                n_steps=n_steps,
                batch_size=int((num_envs * n_steps) // 5),
                n_epochs=10,
                learning_rate=CoScheduler(warmup=int(total_timesteps // 20),
                                          stable_warmup=False,
                                          floor_learning_rate=1e-7,
                                          min_learning_rate=2e-6,
                                          learning_rate=4.5e-3,
                                          total_epochs=total_timesteps,
                                          epsilon=1,
                                          pre_warmup_coef=0.03
                                          )(),
                ent_coef=0.01,
                gamma=0.99,
                stats_window_size=200,
                seed=seed)

    # Train the model
    print(f'Train PPO model with #{num_envs} environments. Using cpu_max-1 or 1 processor(s)')

    model.learn(total_timesteps=20_000_000, progress_bar=True, tb_log_name=f"{main_path}/TB/{env_id}/{exp_id}")

    # Save the model
    model.save(path=path_filename)

    # Changing render mode
    env_kwargs.update(dict(render_mode="rgb_array"))

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
