"""
Main Agent for DCGAN
"""
import numpy as np

from tqdm import tqdm
import shutil
import random

import torch
from torch import nn
from torch.backends import cudnn
from torch.autograd import Variable
import torchvision.utils as vutils

from agents.base import BaseAgent
from graphs.models.dcgan_generator import Generator
from graphs.models.dcgan_discriminator import Discriminator
from graphs.losses.bce import BinaryCrossEntropy
from datasets.celebA import CelebADataLoader

from tensorboardX import SummaryWriter
from utils.metrics import AverageMeter, AverageMeterList
from utils.misc import print_cuda_statistics

cudnn.benchmark = True


class DCGANAgent(BaseAgent):

    def __init__(self, config):
        super().__init__(config)
        # define models ( generator and discriminator)
        self.netG = Generator(self.config)
        self.netD = Discriminator(self.config)
        # define dataloader
        self.dataloader = CelebADataLoader(self.config)

        # define loss
        self.loss = BinaryCrossEntropy()

        # define optimizers for both generator and discriminator
        self.optimG = torch.optim.Adam(self.netG.parameters(), lr=self.config.learning_rate, betas=(self.config.beta1, self.config.beta2))
        self.optimD = torch.optim.Adam(self.netD.parameters(), lr=self.config.learning_rate, betas=(self.config.beta1, self.config.beta2))

        # initialize counter
        self.current_epoch = 0
        self.current_iteration = 0
        self.best_valid_mean_iou = 0

        self.fixed_noise = Variable(torch.randn(self.config.batch_size, self.config.g_input_size, 1, 1))
        self.real_label = 1
        self.fake_label = 0

        # set cuda flag
        self.is_cuda = torch.cuda.is_available()
        if self.is_cuda and not self.config.cuda:
            self.logger.info("WARNING: You have a CUDA device, so you should probably enable CUDA")

        self.cuda = self.is_cuda & self.config.cuda
        # set the manual seed for torch
        #if not self.config.seed:
        self.manual_seed = random.randint(1, 10000)
        #self.manual_seed = self.config.seed
        self.logger.info ("seed: " , self.manual_seed)
        random.seed(self.manual_seed)
        if self.cuda:
            self.device = torch.device("cuda")
            torch.cuda.set_device(self.config.gpu_device)
            torch.cuda.manual_seed_all(self.manual_seed)
            self.logger.info("Program will run on *****GPU-CUDA***** ")
            print_cuda_statistics()
        else:
            self.device = torch.device("cpu")
            torch.manual_seed(self.manual_seed)
            self.logger.info("Program will run on *****CPU***** ")

        self.netG = self.netG.to(self.device)
        self.netD = self.netD.to(self.device)
        self.loss = self.loss.to(self.device)
        self.fixed_noise = self.fixed_noise.to(self.device)
        # Model Loading from the latest checkpoint if not found start from scratch.
        self.load_checkpoint(self.config.checkpoint_file)

        # Summary Writer
        self.summary_writer = SummaryWriter(log_dir=self.config.summary_dir, comment='DCGAN')

    def load_checkpoint(self, file_name):
        filename = self.config.checkpoint_dir + file_name
        try:
            self.logger.info("Loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)

            self.current_epoch = checkpoint['epoch']
            self.current_iteration = checkpoint['iteration']
            self.netG.load_state_dict(checkpoint['G_state_dict'])
            self.optimG.load_state_dict(checkpoint['G_optimizer'])
            self.netD.load_state_dict(checkpoint['D_state_dict'])
            self.optimD.load_state_dict(checkpoint['D_optimizer'])
            self.fixed_noise = checkpoint['fixed_noise']
            self.manual_seed = checkpoint['manual_seed']

            self.logger.info("Checkpoint loaded successfully from '{}' at (epoch {}) at (iteration {})\n"
                  .format(self.config.checkpoint_dir, checkpoint['epoch'], checkpoint['iteration']))
        except OSError as e:
            self.logger.info("No checkpoint exists from '{}'. Skipping...".format(self.config.checkpoint_dir))
            self.logger.info("**First time to train**")

    def save_checkpoint(self, file_name="checkpoint.pth.tar", is_best = 0):
        state = {
            'epoch': self.current_epoch,
            'iteration': self.current_iteration,
            'G_state_dict': self.netG.state_dict(),
            'G_optimizer': self.optimG.state_dict(),
            'D_state_dict': self.netD.state_dict(),
            'D_optimizer': self.optimD.state_dict(),
            'fixed_noise': self.fixed_noise,
            'manual_seed': self.manual_seed
        }
        # Save the state
        torch.save(state, self.config.checkpoint_dir + file_name)
        # If it is the best copy it to another file 'model_best.pth.tar'
        if is_best:
            shutil.copyfile(self.config.checkpoint_dir + file_name,
                            self.config.checkpoint_dir + 'model_best.pth.tar')

    def run(self):
        """
        This function will the operator
        :return:
        """
        try:
            self.train()

        except KeyboardInterrupt:
            self.logger.info("You have entered CTRL+C.. Wait to finalize")

    def train(self):
        for epoch in range(self.current_epoch, self.config.max_epoch):
            self.current_epoch = epoch
            self.train_one_epoch()
            self.save_checkpoint()

    def train_one_epoch(self):
        # initialize tqdm batch
        tqdm_batch = tqdm(self.dataloader.loader, total=self.dataloader.num_iterations, desc="epoch-{}-".format(self.current_epoch))

        self.netG.train()
        self.netD.train()

        epoch_lossG = AverageMeter()
        epoch_lossD = AverageMeter()


        for curr_it, x in enumerate(tqdm_batch):
            #y = torch.full((self.batch_size,), self.real_label)
            x = x[0]
            y = torch.randn(x.size(0), )
            fake_noise = torch.randn(x.size(0), self.config.g_input_size, 1, 1)

            if self.cuda:
                x = x.cuda(async=self.config.async_loading)
                y = y.cuda(async=self.config.async_loading)
                fake_noise = fake_noise.cuda(async=self.config.async_loading)

            x = Variable(x)
            y = Variable(y)
            fake_noise = Variable(fake_noise)
            ####################
            # Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            # train with real
            self.netD.zero_grad()
            D_real_out = self.netD(x)
            y.fill_(self.real_label)
            loss_D_real = self.loss(D_real_out, y)
            loss_D_real.backward()

            # train with fake
            G_fake_out = self.netG(fake_noise)
            y.fill_(self.fake_label)

            D_fake_out = self.netD(G_fake_out.detach())

            loss_D_fake = self.loss(D_fake_out, y)
            loss_D_fake.backward()
            #D_mean_fake_out = D_fake_out.mean().item()

            loss_D = loss_D_fake + loss_D_real
            self.optimD.step()

            ####################
            # Update G network: maximize log(D(G(z)))
            self.netG.zero_grad()
            y.fill_(self.real_label)
            D_out = self.netD(G_fake_out)
            loss_G = self.loss(D_out, y)
            loss_G.backward()

            #D_G_mean_out = D_out.mean().item()

            self.optimG.step()

            epoch_lossD.update(loss_D.item())
            epoch_lossG.update(loss_G.item())

            self.current_iteration += 1

            self.summary_writer.add_scalar("epoch/Generator_loss", epoch_lossG.val, self.current_iteration)
            self.summary_writer.add_scalar("epoch/Discriminator_loss", epoch_lossD.val, self.current_iteration)

        gen_out = self.netG(self.fixed_noise)
        out_img = self.dataloader.plot_samples_per_epoch(gen_out.data, self.current_iteration)
        self.summary_writer.add_image('train/generated_image', out_img, self.current_iteration)

        tqdm_batch.close()

        self.logger.info("Training at epoch-" + str(self.current_epoch) + " | " + "Discriminator loss: " + str(
            epoch_lossD.val) + " - Generator Loss-: " + str(epoch_lossG.val))

    def validate(self):
        pass

    def finalize(self):
        """
        Finalize all the operations of the 2 Main classes of the process the operator and the data loader
        :return:
        """
        self.logger.info("Please wait while finalizing the operation.. Thank you")
        self.save_checkpoint()
        self.summary_writer.export_scalars_to_json("{}all_scalars.json".format(self.config.summary_dir))
        self.summary_writer.close()
        self.dataloader.finalize()
