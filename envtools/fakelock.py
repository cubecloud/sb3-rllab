__version__ = 0.001


class FakeRLock:
    def acquire(self, *args, **kwargs):
        pass

    def release(self, *args, **kwargs):
        pass

    def __enter__(self):
        pass

    def __exit__(self, *args, **kwargs):
        pass