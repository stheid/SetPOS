import time


class stopwatch():
    def __init__(self, verbose=True, prefix=None):
        self.verbose = verbose
        self.prefix = prefix or ''

    @property
    def elapsed(self):
        if hasattr(self, '_interval'):
            return self._interval
        return time.perf_counter() - self.start

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self._interval = self.end - self.start
        if self.verbose:
            if self.prefix:
                print(self.prefix, end=' ')
            print(f'elapsed time: {self.elapsed:.1f}s')
