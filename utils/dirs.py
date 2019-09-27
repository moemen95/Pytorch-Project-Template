from pathlib import Path

from utils.misc import get_datetime_str

_TBOARD_PATH = Path('tboard_logs')
_EXPERIMENTS_PATH = Path('experiments')
_CHECKPOINTS_PATH = Path('checkpoints')
_OUT_PATH = Path('out')
_LOGS_PATH = Path('logs')


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


def get_tensorboard_dir(exp_name, with_timestamp=True):
    return _get_timestamped_path(
        parent_path=_TBOARD_PATH,
        middle_path=exp_name,
        with_timestamp=with_timestamp,
    )


def get_checkpoint_dir(exp_name, with_timestamp=True):
    return _get_timestamped_path(
        parent_path=_EXPERIMENTS_PATH,
        middle_path=exp_name,
        child_path=_CHECKPOINTS_PATH,
        with_timestamp=with_timestamp,
    )


def get_out_dir(exp_name, with_timestamp=True):
    return _get_timestamped_path(
        parent_path=_EXPERIMENTS_PATH,
        middle_path=exp_name,
        child_path=_OUT_PATH,
        with_timestamp=with_timestamp,
    )


def get_logs_dir(exp_name, with_timestamp=True):
    return _get_timestamped_path(
        parent_path=_EXPERIMENTS_PATH,
        middle_path=exp_name,
        child_path=_LOGS_PATH,
        with_timestamp=with_timestamp,
    )
