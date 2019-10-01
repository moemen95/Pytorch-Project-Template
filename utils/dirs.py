from pathlib import Path

import gin

from utils.misc import get_datetime_str

_TBOARD_PATH = Path('tboard_logs')
_EXPERIMENTS_PATH = Path('experiments')
_CHECKPOINTS_PATH = Path('checkpoints')
_OUT_PATH = Path('out')
_LOGS_PATH = Path('logs')

CHECKPOINTS_DIR_GIN_MACRO_NAME = 'checkpoints_dir'

def _get_timestamped_path(
        parent_path,
        middle_path,
        child_path=None,
        with_timestamp=True,
        create=True,
):
    """
    Creates a path to a directory by the following pattern:
     parent_path / middle_path / [child_path] / [timestamp]
    :param parent_path:
    :param middle_path:
    :param child_path:
    :param with_timestamp: if True (default) will append an additional directory at the end of the path
        with a timedate string
    :param create: if True (default) will create all the directories in the path
    :return: pathlib.Path object
    """
    dir_path = parent_path / middle_path
    dir_path = dir_path / child_path if child_path is not None else dir_path

    if with_timestamp:
        dir_path /= get_datetime_str()

    if create:
        dir_path.mkdir(parents=True, exist_ok=True)

    return dir_path


@gin.configurable
def make_exp_dirs(exp_name):
    return (
        # tboard
        _get_timestamped_path(
            parent_path=_TBOARD_PATH,
            middle_path=exp_name,
        ),
        # checkpoints
        _get_timestamped_path(
            parent_path=_EXPERIMENTS_PATH,
            middle_path=exp_name,
            child_path=_CHECKPOINTS_PATH,
        ),
        # out dir
        _get_timestamped_path(
            parent_path=_EXPERIMENTS_PATH,
            middle_path=exp_name,
            child_path=_OUT_PATH,
        ),
        # logs
        _get_timestamped_path(
            parent_path=_EXPERIMENTS_PATH,
            middle_path=exp_name,
            child_path=_LOGS_PATH,
        ),
    )
