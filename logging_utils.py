import time
import logging
import numpy as np

from typing import Optional
from logging.handlers import RotatingFileHandler

from general import txt


def get_logger(name=__name__, logfile="logfile.log"):
    logging.basicConfig(
         filename=logfile,
         level=logging.DEBUG,
         format= '[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
         datefmt='%H:%M:%S'
     )

    handler = RotatingFileHandler(logfile, maxBytes=1024*1024*4, backupCount=1)
    logging.getLogger('').addHandler(handler)
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    # set a format which is simpler for console use
    formatter = logging.Formatter('[%(name)-12s] %(levelname)-8s| %(asctime)s | %(message)s')
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)
    return logging.getLogger(name)


class TimedBlock:
    items = dict()

    def __init__(self, name, supress=True):
        self.t0 = time.time()
        self.name = name
        self.suppressed = supress

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.t0
        if not self.suppressed:
            print(f"{self.name} | elapsed: {elapsed:.4f}s")

        samples = TimedBlock.items.get(self.name, [])
        samples.append(elapsed)
        TimedBlock.items[self.name] = samples

    @classmethod
    def stats(cls, name: Optional[str] = None):
        if name is None:
            names = TimedBlock.items.keys()
        else:
            if name not in TimedBlock.items:
                return
            else:
                names = [name]

        values = []
        for name in names:
            samples = TimedBlock.items[name]
            mean = np.mean(samples)
            std = np.std(samples)
            total = sum(samples)
            num = len(samples)
            values.append((total, name, num, mean, std))

        values = sorted(values, key=lambda x: x[0])
        for total, name, num, mean, std in values:
            exp_value = int(np.floor(np.log(mean) / np.log(1000)))
            mean_formatted = f"{mean * 1000 ** (-exp_value):.2f} · 10^{exp_value * 3}"
            std_normal = std / mean * 100
            logger.debug(f"{name} | mean: {mean_formatted}s (+- {std_normal:.2f}%) | # {num} | total: {total:.4f}s")


if __name__ == "__main__":
    logger = get_logger()
    logger.debug(txt("Hello %r  Friend!%.  how are you? What '%%' are you at? %by Actually,"))
    logger.debug(txt("%.. Heck"))
    logger.debug(txt("Hello %rkiWorld"))
    logger.debug(txt("Hello %rkbWorld%.k., how are you?"))
    logger.debug(txt("Hello %rkiWorld"))
