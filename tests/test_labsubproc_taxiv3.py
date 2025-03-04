import os
import logging
import datetime

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

from pytz import timezone
from multiprocessing import freeze_support
from multiprocessing import get_logger

from sb3_rllab import LabSubprocVecEnv
from sb3_rllab import CoScheduler
from sb3_rllab import add_score, save_mp4

logger = get_logger()

TZ = timezone("Europe/Moscow")

if __name__ == "__main__":
    freeze_support()

    file_handler = logging.FileHandler("test_labsubprocvecenv_recurrent_ppo.log")
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    seed = 42
    num_envs = 280
    total_timesteps = 30_000_000

    env_id = "Taxi-v3"
    env_kwargs = dict(render_mode=None,  # render off
                      # continuous=True,
                      # gravity=-9.8,
                      # enable_wind=True,
                      # wind_power=15.0,
                      # turbulence_power=1.5
                      )

    train_vec_env_kwargs = dict(env_id=env_id,
                                env_kwargs=env_kwargs,
                                n_envs=num_envs,
                                seed=seed,
                                vec_env_cls=LabSubprocVecEnv,
                                vec_env_kwargs=dict(use_threads=False,
                                                    n_processes=None,
                                                    use_period="train")
                                )

    eval_vec_env_kwargs = dict(env_id=env_id,
                               env_kwargs=env_kwargs,
                               n_envs=28,
                               seed=42,
                               vec_env_cls=LabSubprocVecEnv,
                               vec_env_kwargs=dict(use_threads=False,
                                                   n_processes=None,
                                                   use_period="test")
                               )

    train_vec_env = make_vec_env(**train_vec_env_kwargs)

    exp_id = f'{datetime.datetime.now(TZ).strftime("%y%d%m-%H%M%S")}'
    main_path = "."
    n_steps = 1000
    model = DQN("MlpPolicy",
                train_vec_env,
                verbose=1,
                device="auto",
                batch_size=int((num_envs * n_steps) // 5),
                learning_rate=CoScheduler(warmup=int(total_timesteps // 20),
                                          stable_warmup=False,
                                          floor_learning_rate=1e-6,
                                          min_learning_rate=1e-5,
                                          learning_rate=1e-3,
                                          total_epochs=total_timesteps,
                                          epsilon=1,
                                          pre_warmup_coef=0.03
                                          )(),
                gamma=0.99,
                stats_window_size=n_steps,
                seed=seed,
                tensorboard_log=os.path.join(main_path, "TB"),
                )

    policy_id = f'{model.__class__.__name__}'
    path_filename = os.path.join(main_path, f"{policy_id}_{env_id}")

    eval_env = make_vec_env(**eval_vec_env_kwargs)  # Create a separate environment for evaluation
    eval_callback = EvalCallback(eval_env,
                                 best_model_save_path=os.path.join(path_filename, 'best_model'),
                                 # log_path='./logs',
                                 eval_freq=int(total_timesteps // 10 // num_envs),
                                 n_eval_episodes=28,
                                 deterministic=False,
                                 render=False)

    # Learn the model
    model.learn(total_timesteps=total_timesteps, progress_bar=True, callback=eval_callback,
                tb_log_name=f"{policy_id}_{env_id}_{exp_id}")

    # Changing VecEnv to DummyVecEnv and render mode
    eval_vec_env_kwargs = dict(env_id=env_id,
                               env_kwargs=env_kwargs,
                               n_envs=1,
                               seed=42,
                               vec_env_cls=DummyVecEnv,
                               )
    # noinspection PyTypeChecker
    env_kwargs.update(dict(render_mode="rgb_array"))

    # Load the model
    eval_vec_env = make_vec_env(**eval_vec_env_kwargs)
    model = DQN.load(path=os.path.join(path_filename, 'best_model', 'best_model.zip'), env=eval_vec_env, device="cpu")
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
