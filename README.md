# SB3-RLLab

**Custom tools for Stable Baselines 3 Reinforcement Learning Laboratory**

## Description

This project provides a set of tools for working with the **Stable Baselines 3** library, which is widely used in reinforcement learning tasks. The tools include improved versions of standard components from the library as well as additional utilities for managing processes and multithreading.

## Main Tools

### 1. LabSubprocVecEnv

An enhanced version of the standard `SubProcVecEnv` class that allows you to run thousands of environments simultaneously in multiprocessing mode. It also supports the use of `threading.RLock()` in threading mode (see the next tool).

### 2. SMpLock and SThLock

Singleton locks designed for use with various custom environments and scenarios. These locks can be used together with `LabSubprocVecEnv` in threading mode. They can be applied with the `unique_name` parameter.

### 3. FakeLock

A dummy (stub) lock that can be used when you need to create a placeholder instead of a real lock.

### 4. Learning Rate Scheduler

A utility for managing the learning rate during model training. Example usage:

```python
total_timesteps = 20_000_000
learning_rate = CoScheduler(
    warmup=int(total_timesteps // 20),
    stable_warmup=False,
    floor_learning_rate=1e-7,
    min_learning_rate=2e-6,
    learning_rate=4.5e-6,
    total_epochs=total_timesteps,
    epsilon=1,
    pre_warmup_coef=0.03
)()
```
<video width="320" height="240" controls>
    
  <source src=https://github.com/cubecloud/sb3-rllab/main/tests/LunarLander-v2_250702-185132.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>
