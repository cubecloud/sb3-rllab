import os
import logging
import datetime
import numpy as np

from sb3_contrib import RecurrentPPO
from gymnasium.spaces import Box
# from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import StackedObservations
from stable_baselines3.common.vec_env import VecFrameStack
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

    seed = 443
    num_envs = 280
    total_timesteps = 60_000_000
    lookback = 20
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
    train_vec_env = VecFrameStack(train_vec_env, n_stack=lookback, channels_order='last')

    exp_id = f'{datetime.datetime.now(TZ).strftime("%y%d%m-%H%M%S")}'
    main_path = "."

    n_steps = 200
    model = RecurrentPPO("MlpLstmPolicy",
                         train_vec_env,
                         verbose=1,
                         device="auto",
                         n_steps=n_steps,
                         batch_size=int((num_envs * n_steps) // 5),
                         n_epochs=10,
                         learning_rate=CoScheduler(warmup=int(total_timesteps // 20),
                                                   stable_warmup=False,
                                                   floor_learning_rate=1e-6,
                                                   min_learning_rate=5e-5,
                                                   learning_rate=5e-4,
                                                   total_epochs=total_timesteps,
                                                   epsilon=1,
                                                   pre_warmup_coef=0.1
                                                   )(),
                         max_grad_norm=0.25,
                         ent_coef=0.01,
                         gamma=0.95,
                         stats_window_size=n_steps,
                         seed=seed,
                         tensorboard_log=os.path.join(main_path, "TB"),
                         )

    policy_id = f'{model.__class__.__name__}'
    path_filename = os.path.join(main_path, f"{policy_id}_{env_id}")

    eval_env = make_vec_env(**eval_vec_env_kwargs)  # Create a separate environment for evaluation
    eval_env = VecFrameStack(eval_env, n_stack=lookback, channels_order='last')
    eval_callback = EvalCallback(eval_env,
                                 best_model_save_path=os.path.join(path_filename, 'best_model'),
                                 # log_path='./logs',
                                 eval_freq=int(1_000_000//num_envs),
                                 n_eval_episodes=28,
                                 deterministic=False,
                                 render=False)
    # Deterministic is used to control the stochasticity of actions in the evaluation environment.
    # If it's set to True, the actions are chosen based on argmax of Q-values.
    # Otherwise, a random action is chosen from multinomial distribution.

    # Learn the model
    model.learn(total_timesteps=total_timesteps, progress_bar=True, callback=eval_callback,
                tb_log_name=os.path.join(f"{policy_id}_{env_id}_{exp_id}"))

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
    eval_vec_env = VecFrameStack(eval_vec_env, n_stack=lookback, channels_order='last')
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
