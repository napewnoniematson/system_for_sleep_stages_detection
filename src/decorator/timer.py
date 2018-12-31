import time
import src.logger.logger as logger


def timer(method):
    def timed(*args, **kwargs):
        ts = time.time()
        result = method(*args, **kwargs)
        te = time.time()
        time_diff = te - ts
        logger.info("Method: {0} finished with time {1} seconds".format(
            method.__name__, time_diff
        ))
        return result

    return timed
