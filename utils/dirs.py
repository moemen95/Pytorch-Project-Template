from pathlib import Path

import gin

from utils.misc import get_datetime_str

_TBOARD_PATH = Path('tboard_logs')
_EXPERIMENTS_PATH = Path('experiments')
_CHECKPOINTS_PATH = Path('checkpoints')
_OUT_PATH = Path('out')
_LOGS_PATH = Path('logs')

CHECKPOINTS_DIR_GIN_MACRO_NAME = 'checkpoints_dir'
TBOARD_DIR_GIN_MACRO_NAME = 'tboard_dir'


def _get_timestamped_path(
        parent_path: Path,
        middle_path,
        child_path='',
        timestamp='',
        create=True,
):
    """
    Creates a path to a directory by the following pattern:
     parent_path / middle_path / [child_path] / [timestamp]
    :param parent_path:
    :param middle_path:
    :param child_path:
    :param timestamp
    :param create: if True (default) will create all the directories in the path
    :return: pathlib.Path object
    """
    dir_path = parent_path / middle_path / child_path / timestamp

    if create:
        dir_path.mkdir(parents=True, exist_ok=True)

    return dir_path


@gin.configurable
def make_exp_dirs(exp_name):
    timestamp = get_datetime_str()

    return (
        # tboard
        _get_timestamped_path(
            parent_path=_TBOARD_PATH,
            middle_path=exp_name,
            timestamp=timestamp,
        ),
        # checkpoints
        _get_timestamped_path(
            parent_path=_EXPERIMENTS_PATH,
            middle_path=exp_name,
            child_path=_CHECKPOINTS_PATH,
            timestamp=timestamp,
        ),
        # out dir
        _get_timestamped_path(
            parent_path=_EXPERIMENTS_PATH,
            middle_path=exp_name,
            child_path=_OUT_PATH,
            timestamp=timestamp,
        ),
        # logs
        _get_timestamped_path(
            parent_path=_EXPERIMENTS_PATH,
            middle_path=exp_name,
            child_path=_LOGS_PATH,
            timestamp=timestamp,
        ),
    )


def get_current_tboard_dir():
    return gin.query_parameter(f'%{TBOARD_DIR_GIN_MACRO_NAME}')


def get_current_checkpoints_dir():
    return gin.query_parameter(f'%{CHECKPOINTS_DIR_GIN_MACRO_NAME}')
