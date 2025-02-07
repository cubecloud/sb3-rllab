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


if __name__ == "__main__":
    print('Check test')
    import multiprocessing
    obj1 = SMpLock(lock=multiprocessing.Lock(), unique_name='one')
    obj2 = SMpLock(lock=multiprocessing.Lock(), unique_name='two')
    obj3 = SMpLock(lock=multiprocessing.Lock(), unique_name='three')
    obj4 = SMpLock(lock=multiprocessing.Lock(), unique_name='three')

    print(id(obj1))  # prints the address of obj1
    print('Lock', id(obj1.lock))  # prints the address of obj1.lock
    print(id(obj2))  # prints the address of obj2
    print('Lock', id(obj2.lock))  # prints the address of obj1.lock
    print(id(obj3))  # prints the address of obj3
    print('Lock', id(obj3.lock))  # prints the address of obj1.lock
    print(id(obj4))  # prints the address of obj3
    print('Lock', id(obj4.lock))  # prints the address of obj1.lock
