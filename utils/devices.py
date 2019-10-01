import logging

import gin
import torch
import numpy as np

from utils.misc import print_cuda_statistics

logger = logging.getLogger()


@gin.configurable
def configure_device(cuda, gpu_device=None, seed=None):
    """Wrapper function to call _set_device and _set_random_seed functions"""

    cuda_available = torch.cuda.is_available()
    if cuda_available and not cuda:
        logger.info("WARNING: You have a CUDA device, but chose not to use it.")
    if not cuda_available and cuda:
        logger.info("WARNING: You specified cuda=True but no CUDA device found.")

    use_cuda = cuda_available & cuda

    device = _set_device(use_cuda=use_cuda, gpu_device=gpu_device)
    _set_random_seed(use_cuda=use_cuda, seed=seed)

    return device


def _set_device(use_cuda, gpu_device):
    if use_cuda:
        device = torch.device("cuda")
        torch.cuda.device(gpu_device)
        logger.info("Program will run on *****GPU-CUDA***** ")
        print_cuda_statistics()
    else:
        device = torch.device("cpu")
        logger.info("Program will run on *****CPU*****\n")

    return device


def _set_random_seed(use_cuda, seed=None):
    """
    See https://pytorch.org/docs/stable/notes/randomness.html
    and https://stackoverflow.com/questions/55097671/how-to-save-and-load-random-number-generator-state-in-pytorch
    """

    if seed is not None:
        logger.info(f'Setting manual seed = {seed}')
        torch.manual_seed(seed)
        np.random.seed(seed)

        # even when setting the random seed cuda devices can behave nondeterministically
        # setting those flags reduces this nondeterminism
        if use_cuda:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
