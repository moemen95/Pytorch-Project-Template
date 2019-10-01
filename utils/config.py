import logging
from logging import Formatter
from logging.handlers import RotatingFileHandler

import gin

from utils.dirs import make_exp_dirs, CHECKPOINTS_DIR_GIN_MACRO_NAME, TBOARD_DIR_GIN_MACRO_NAME


def setup_logging(log_dir):
    log_file_format = "[%(levelname)s] - %(asctime)s - %(name)s - : %(message)s in %(pathname)s:%(lineno)d"
    log_console_format = "[%(levelname)s]: %(message)s"

    # Main logger
    main_logger = logging.getLogger()
    main_logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(Formatter(log_console_format))

    exp_file_handler = RotatingFileHandler('{}exp_debug.log'.format(log_dir), maxBytes=10 ** 6, backupCount=5)
    exp_file_handler.setLevel(logging.DEBUG)
    exp_file_handler.setFormatter(Formatter(log_file_format))

    exp_errors_file_handler = RotatingFileHandler('{}exp_error.log'.format(log_dir), maxBytes=10 ** 6, backupCount=5)
    exp_errors_file_handler.setLevel(logging.WARNING)
    exp_errors_file_handler.setFormatter(Formatter(log_file_format))

    main_logger.removeHandler(main_logger.handlers[0])
    main_logger.addHandler(console_handler)
    main_logger.addHandler(exp_file_handler)
    main_logger.addHandler(exp_errors_file_handler)


def _gin_add_kwargs(gin_kwargs: dict):
    """Updates the gin config by adding the passed values as gin macros."""
    for key, val in gin_kwargs.items():
        gin.bind_parameter(binding_key=f'%{key}', value=val)


def process_gin_config(config_file, gin_kwargs: dict):
    # add custom values not provided in the config file as macros
    _gin_add_kwargs(gin_kwargs)

    gin.parse_config_file(config_file=config_file)

    # create some important directories to be used for that experiment
    summary_dir, checkpoints_dir, out_dir, log_dir = make_exp_dirs(exp_name=gin.REQUIRED)
    _gin_add_kwargs({
        CHECKPOINTS_DIR_GIN_MACRO_NAME: checkpoints_dir,
        TBOARD_DIR_GIN_MACRO_NAME: summary_dir,
    })

    # setup logging in the project
    setup_logging(log_dir)
    logger = logging.getLogger()

    logger.info(f"The experiment name is '{gin.query_parameter('%exp_name')}'")
    logger.info("Configuration:")
    logging.info(gin.config.config_str())

    logger.info("Configurations are successfully processed and dirs are created.")
    logger.info("The pipeline of the project will begin now.")
