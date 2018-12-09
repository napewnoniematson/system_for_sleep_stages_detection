import time


def timer(method):
    def timed(*args, **kwargs):
        ts = time.time()
        result = method(*args, **kwargs)
        te = time.time()
        time_diff = te - ts
        print("timeit")
        return result, time_diff

    return timed
