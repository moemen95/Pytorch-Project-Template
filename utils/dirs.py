import os
import logging
from pathlib import Path

from utils.misc import get_datetime_str

TBOARD_PATH = Path('tboard_logs')


def make_tensorboard_dir(exp_name, with_timestamp=True):
    tboard_dir = TBOARD_PATH / exp_name
    if with_timestamp:
        tboard_dir /= get_datetime_str()

    return tboard_dir


def create_dirs(dirs):
    """
    dirs - a list of directories to create if these directories are not found
    :param dirs:
    :return:
    """
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
    except Exception as err:
        logging.getLogger("Dirs Creator").info("Creating directories error: {0}".format(err))
        exit(-1)
