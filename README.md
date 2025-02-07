# sb3-rllab
Stable Baselines 3 reinforcement learning laboratory custom tools

## Tools
1. LabSubprocVecEnv - enhanced version of SubProcVecEnv, can running thousands environments in each process in multiprocessing mode, can use threading.RLock() in threading mode check next tool
2. SMpLock, SThLock - singleton Locks for using with different custom environments an situations, created to using with LabSubprocVecEnv in threading mode. Can be used with 'unique_name'
3. FakeLock - dummy (stub) lock

