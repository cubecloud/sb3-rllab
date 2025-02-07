from typing import Callable
from math import cos, pi

__version__ = 0.008


class CoScheduler:
    def __init__(self, warmup: int = 15, learning_rate: float = 1e-4, min_learning_rate: float = 1e-6,
                 total_epochs: int = 300, epsilon: int = 1, stable_warmup: bool = False,
                 floor_learning_rate: float = 1e-6, pre_warmup_coef: float = 0.03):
        """
            Scheduler for learning rate for standard_baselines3
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
        Args:
            warmup (int):                   warmup timesteps
            learning_rate (float):          base learning rate
            min_learning_rate:              min learning rate (where we going at the end)
            total_epochs(int):              total timesteps (epochs)
            epsilon (int):                  how often recalculate learning_rate (timesteps)
            stable_warmup (bool):           all warmup timesteps will be on min_learning_rate
            floor_learning_rate (float):    floor-learning_rate (especially created for relearning models)
            pre_warmup_coef (float):        The pre_warmup coefficient is calculated based on the proportion
                                            of warmup steps, with a default value of 3% of the warmup timesteps.
        """
        self.warmup = warmup
        self.stable_warmup = stable_warmup
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.total_epochs = total_epochs
        self.last_lr = self.min_learning_rate
        self.epsilon = epsilon
        """
        pre_warmup - using 'pre_warmup_coef' (1/3) of warmup time with floor_learning_rate, 
        for setup RL system properly on 'road'
        """
        self.floor_learning_rate = floor_learning_rate
        self.pre_warmup: int = int(self.warmup * pre_warmup_coef)
        self.__call__()

    def __call__(self) -> Callable[[float], float]:
        return self.scheduler

    def scheduler(self, progress_remaining: float) -> float:
        epoch = int((1 - progress_remaining) * self.total_epochs)
        if epoch % self.epsilon == 0:
            """ Warm up from zero to learning_rate """
            if epoch < self.warmup:
                if epoch < self.pre_warmup:
                    lr = self.floor_learning_rate
                else:
                    if self.stable_warmup:
                        lr = self.min_learning_rate
                    else:
                        # exponential increasing learning_rate
                        lr = self.min_learning_rate * (self.learning_rate / self.min_learning_rate) ** (
                                (epoch - self.pre_warmup) / (self.warmup - self.pre_warmup))
            else:
                """ using cos learning rate """
                formula = self.min_learning_rate + 0.5 * (self.learning_rate - self.min_learning_rate) * (
                        1 + cos(max(epoch + 1 - self.warmup, 0) * pi / max(self.total_epochs - self.warmup, 1)))
                # min calc min and max with zero centered logic - close to zero = less
                lr = max(formula, self.min_learning_rate)
            self.last_lr = lr
        return self.last_lr
