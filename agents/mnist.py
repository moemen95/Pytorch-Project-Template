"""
Mnist Main agent, as mentioned in the tutorial
"""

import torch
from torch.backends import cudnn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from agents.base import BaseAgent
from graphs.models.mnist import Mnist
from datasets.mnist import MnistDataLoader
from utils.misc import print_cuda_statistics

cudnn.benchmark = True


class MnistAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)

        # set agent name
        self.agent_name = 'MNIST'
        # define models
        self.model = Mnist()
        # define data_loader
        self.data_loader = MnistDataLoader(config=config)
        # define loss
        self.loss_fn = F.nll_loss

        # define optimizer
        self.optimizer = optim.SGD(
            params=self.model.parameters(),
            lr=self.config.learning_rate,
            momentum=self.config.momentum,
        )

        # initialize counter
        self.current_epoch = 0
        self.current_iteration = 0
        self.best_metric = 0

        # set cuda flag
        self.is_cuda = torch.cuda.is_available()
        if self.is_cuda and not self.config.cuda:
            self.logger.info("WARNING: You have a CUDA device, so you should probably enable CUDA")

        self.cuda = self.is_cuda & self.config.cuda

        # set the manual seed for torch
        self.manual_seed = self.config.seed
        if self.cuda:
            torch.cuda.manual_seed(self.manual_seed)
            self.device = torch.device("cuda")
            torch.cuda.set_device(self.config.gpu_device)
            self.model = self.model.to(self.device)

            self.logger.info("Program will run on *****GPU-CUDA***** ")
            print_cuda_statistics()
        else:
            self.device = torch.device("cpu")
            torch.manual_seed(self.manual_seed)
            self.logger.info("Program will run on *****CPU*****\n")

        # Model Loading from the latest checkpoint if not found start from scratch.
        self.load_checkpoint(self.config.checkpoint_file)
        # Summary Writer
        self.summary_writer = SummaryWriter(log_dir=self.config.summary_dir, comment=self.agent_name)

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
            self.logger.info("You have entered CTRL+C.. Wait to finalize")

    def train(self):
        """
        Main training loop
        :return:
        """
        for epoch in range(1, self.config.max_epoch + 1):
            self.train_one_epoch()
            self.validate()

            self.current_epoch += 1

    def train_one_epoch(self):
        """
        One epoch of training
        :return:
        """
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.data_loader.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss_fn(output, target)
            loss.backward()
            self.optimizer.step()
            if batch_idx % self.config.log_interval == 0:
                loss_val = loss.item()
                self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    self.current_epoch, batch_idx * len(data),
                    len(self.data_loader.train_loader.dataset),
                    100. * batch_idx / len(self.data_loader.train_loader),
                    loss_val,
                ))
                # log to tensorboard
                self.summary_writer.add_scalar(
                    tag='Loss/train',
                    scalar_value=loss_val,
                    global_step=self.current_iteration,
                )
            self.current_iteration += 1

    def validate(self):
        """
        One cycle of model validation
        :return:
        """
        self.model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.data_loader.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                # sum up batch loss
                val_loss += self.loss_fn(output, target, size_average=False).item()
                # get the index of the max log-probability
                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(target.view_as(pred)).sum().item()

        val_loss /= len(self.data_loader.val_loader.dataset)
        val_accuracy = 100. * correct / len(self.data_loader.val_loader.dataset)

        self.logger.info('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            val_loss,
            correct,
            len(self.data_loader.val_loader.dataset),
            val_accuracy,
        ))
        # log to tensorboard
        self.summary_writer.add_scalar(
            tag='Loss/validation',
            scalar_value=val_loss,
            global_step=self.current_iteration,
        )
        self.summary_writer.add_scalar(
            tag='Accuracy/validation',
            scalar_value=val_accuracy,
            global_step=self.current_iteration,
        )

    def finalize(self):
        """
        Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
        :return:
        """
        pass
