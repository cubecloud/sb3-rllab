# sb3-rllab
Stable Baselines 3 reinforcement learning laboratory custom tools

## Tools
1. LabSubprocVecEnv - enhanced version of SubProcVecEnv, can running thousands environments in each process in multiprocessing mode, can use threading.RLock() in threading mode check next tool
2. SMpLock, SThLock - singleton Locks for using with different custom environments an situations, created to using with LabSubprocVecEnv in threading mode. Can be used with 'unique_name'
3. FakeLock - dummy (stub) lock
4. Scheduler for learning rate for standard_baselines3
   Using example:
   ...
   total_timesteps = 20_000_000
   learning_rate=CoSheduller(warmup=int(total_timesteps // 20),
                 stable_warmup=False,
                 floor_learning_rate=1e-7,
                 min_learning_rate=2e-6,
                 learning_rate=4.5e-6,
                 total_epochs=total_timesteps,
                 epsilon=1,
                 pre_warmup_coef=0.03
                 )()
   ...

