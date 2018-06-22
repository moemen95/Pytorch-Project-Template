import numpy as np

from tqdm import tqdm
import shutil
import random

import torch
from torch import nn
from torch.backends import cudnn
from torch.autograd import Variable
import torchvision.utils as vutils

from graphs.models.generator import Generator
from graphs.models.discriminator import Discriminator
from graphs.losses.loss import BinaryCrossEntropy
from datasets.celebA import CelebADataLoader

from tensorboardX import SummaryWriter
from utils.metrics import AverageMeter, AverageMeterList, evaluate
from utils.misc import print_cuda_statistics

cudnn.benchmark = True

class DCGANAgent:

    def __init__(self, config):
        self.config = config

        # define models
        self.netG = None
        self.netD = None

        # define dataloader
        self.dataloader = None

        # define loss
        self.loss = None

        # define optimizers for both generator and discriminator
        self.optimG = None
        self.optimD = None

        # initialize counter
        self.current_epoch = 0
        self.current_iteration = 0
        self.best_valid_mean_iou = 0

        self.batch_size = None

        self.fixed_noise = None
        self.real_label = 1
        self.fake_label = 0

        # set cuda flag
        self.is_cuda = torch.cuda.is_available()
        if self.is_cuda and not self.config.cuda:
            print("WARNING: You have a CUDA device, so you should probably enable CUDA")

        self.cuda = self.is_cuda & self.config.cuda

        # set the manual seed for torch
        self.manual_seed = None
        if self.cuda:
            print("Program will run on *****GPU-CUDA***** ")
            torch.cuda.manual_seed_all(self.manual_seed)
            print_cuda_statistics()

            self.netG = self.netG.cuda()
            self.netD = self.netD.cuda()
            self.loss = self.loss.cuda()
            self.fixed_noise = self.fixed_noise.cuda(async=self.config.async_loading)

        else:
            print("Program will run on *****CPU***** ")

        # Model Loading from the latest checkpoint if not found start from scratch.
        self.load_checkpoint(self.config.checkpoint_file)
        # Summary Writer
        self.summary_writer = None

    def load_checkpoint(self, file_name):
        """
        Latest checkpoint loader
        :param file_name: name of the checkpoint file
        :return:
        """
        pass

    def save_checkpoint(self, file_name="checkpoint.pth.tar", is_best=0):
        """
        Checkpoint saver
        :param file_name: name of the checkpoint file
        :param is_best: boolean flag to indicate whether current checkpoint's accuracy is the best so far
        :return:
        """
        pass

    def run(self):
        """
        The main operator
        :return:
        """
        try:
            self.train()

        except KeyboardInterrupt:
            print("You have entered CTRL+C.. Wait to finalize")

    def train(self):
        """
        Main training loop
        :return:
        """
        pass

    def train_one_epoch(self):
        """
        One epoch of training
        :return:
        """
        pass

    def validate(self):
        """
        One cycle of model validation
        :return:
        """
        pass

    def finalize(self):
        """
        Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
        :return:
        """
        pass
