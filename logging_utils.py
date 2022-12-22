import logging
from general import txt


def get_logger(name=__name__, logfile="asdf.log"):
    logging.basicConfig(
         filename=logfile,
         level=logging.DEBUG,
         format= '[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
         datefmt='%H:%M:%S'
     )

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    # set a format which is simpler for console use
    formatter = logging.Formatter('[%(name)-12s] %(levelname)-8s| %(asctime)s | %(message)s')
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)
    return logging.getLogger(name)


if __name__ == "__main__":
    logger = get_logger()
    logger.debug(txt("Hello %r  Friend!%.  how are you? What '%%' are you at? %by Actually,"))
    logger.debug(txt("%.. Heck"))
    logger.debug(txt("Hello %rkiWorld"))
    logger.debug(txt("Hello %rkbWorld%.k., how are you?"))
    logger.debug(txt("Hello %rkiWorld"))
