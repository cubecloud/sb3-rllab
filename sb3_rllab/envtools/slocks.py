__version__ = 0.002


class SingletonMpLock(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        unique_name = kwargs.get('unique_name')
        if unique_name is None:
            raise ValueError("unique_name is required")
        if unique_name not in cls._instances:
            cls._instances[unique_name] = super(SingletonMpLock, cls).__call__(*args, **kwargs)
        return cls._instances[unique_name]


class SingletonThLock(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        unique_name = kwargs.get('unique_name')
        if unique_name is None:
            raise ValueError("unique_name is required")
        if unique_name not in cls._instances:
            cls._instances[unique_name] = super(SingletonThLock, cls).__call__(*args, **kwargs)
        return cls._instances[unique_name]


class SMpLock(metaclass=SingletonMpLock):
    def __init__(self, lock: callable, unique_name='train_rlock') -> None:
        self.unique_name = unique_name
        self.lock = lock


class SThLock(metaclass=SingletonThLock):
    def __init__(self, lock: callable, unique_name='train_rlock') -> None:
        self.unique_name = unique_name
        self.lock = lock
