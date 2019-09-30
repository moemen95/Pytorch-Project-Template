"""
The Base Agent class, where all other agents inherit from, that contains definitions for all the necessary functions
"""
import logging

import torch
from torch.utils.tensorboard import SummaryWriter

from utils.misc import print_cuda_statistics


class BaseAgent:
    """
    This base class will contain the base functions to be overloaded by any agent you will implement.
    """

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("Agent")
        self.agent_name = 'Base'

    def _get_state_dict(self):
        return {
            'config': self.config,
        }

    def load_checkpoint(self, file_name):
        """
        Latest checkpoint loader
        :param file_name: name of the checkpoint file
        :return:
        """
        raise NotImplementedError

    def save_checkpoint(self, is_best=False):
        """
        Checkpoint saver
        :param is_best: boolean flag to indicate whether current checkpoint's metric is the best so far
        :return:
        """
        raise NotImplementedError

    def run(self):
        """
        The main operator
        :return:
        """
        raise NotImplementedError

    def train(self):
        """
        Main training loop
        :return:
        """
        raise NotImplementedError

    def train_one_epoch(self):
        """
        One epoch of training
        :return:
        """
        raise NotImplementedError

    def validate(self):
        """
        One cycle of model validation
        :return:
        """
        raise NotImplementedError

    def finalize(self):
        """
        Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
        :return:
        """
        raise NotImplementedError

    @property
    def debug(self):
        return self.config.debug


class BaseTrainAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)

        self._init_counters()
        self._init_model()
        self._init_data_loader()
        self._init_device(use_cuda=self.config.cuda)
        self.set_random_seed(seed=self.config.seed)

        # load model from the latest checkpoint - if not specified start from scratch.
        self.load_checkpoint(self._checkpoint_path)
        # init SummaryWriter for tensorboard logging
        self.summary_writer = SummaryWriter(log_dir=self.config.summary_dir, comment=self.agent_name)

    def _init_counters(self):
        self.current_epoch = 0
        self.current_iteration = 0

    def _init_model(self):
        raise NotImplementedError()

    def _init_data_loader(self):
        raise NotImplementedError()

    def _init_device(self, use_cuda=True):
        cuda_available = torch.cuda.is_available()
        if cuda_available and not use_cuda:
            self.logger.info("WARNING: You have a CUDA device, but chose not to use it.")

        self.cuda = cuda_available & use_cuda

        if self.cuda:
            self.device = torch.device("cuda")
            torch.cuda.set_device(self.config.gpu_device)
            self.logger.info("Program will run on *****GPU-CUDA***** ")
            print_cuda_statistics()
        else:
            self.device = torch.device("cpu")
            self.logger.info("Program will run on *****CPU*****\n")

        self.model = self.model.to(self.device)

    def set_random_seed(self, seed=None):
        self.manual_seed = seed

        if self.manual_seed is not None:
            if self.cuda:
                torch.cuda.manual_seed(self.manual_seed)
            else:
                torch.manual_seed(self.manual_seed)

    def _get_state_dict(self):
        state_dict = super()._get_state_dict()

        state_dict.update({
            'epoch': self.current_epoch,
            'iteration': self.current_iteration,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        })

        return state_dict

    @property
    def _checkpoint_path(self):
        return getattr(self.config, 'checkpoint_file', None)

    def load_checkpoint(self, file_name=None):
        """
        Latest checkpoint loader
        :param file_name: name of the checkpoint file
        :return:
        """
        if file_name is not None and len(file_name) > 0:
            # select the experiment directory - checkpoint_dir is of structure .../exp_name/datetime_str/
            # and we will look for the checkpoint file under a different timestamp
            checkpoint_path = self.config.checkpoint_dir.parent
            checkpoint_path /= file_name

            self.logger.info(f'Loading checkpoint "{checkpoint_path}"')
            checkpoint = torch.load(checkpoint_path)

            self.current_epoch = checkpoint['epoch'] + 1
            self.current_iteration = checkpoint['iteration']
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            self.logger.info(
                f"""Checkpoint loaded successfully from '{self.config.checkpoint_dir}' at (epoch {checkpoint['epoch']}) 
                at (iteration {checkpoint['iteration']})\n""")

    def save_checkpoint(self, is_best=False):
        """
        Checkpoint saver
        :param is_best: boolean flag to indicate whether current checkpoint's accuracy is the best so far
        :return:
        """
        checkpoint_path = self.config.checkpoint_dir / f'epoch_{self.current_epoch}.pth'
        torch.save(self._get_state_dict(), checkpoint_path)
        if is_best:
            best_checkpoint_path = self.config.checkpoint_dir / 'best.pth'
            torch.save(self._get_state_dict(), best_checkpoint_path)

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
        for epoch in range(self.current_epoch, self.config.max_epoch):
            self.train_one_epoch()
            self.validate()
            self.save_checkpoint()

            self.current_epoch += 1

            if self.debug:
                break

    def _log_train_iter(self, batch_idx, loss_val):
        self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            self.current_epoch,
            batch_idx * self.data_loader.train_loader.batch_size,
            self.num_train_samples,
            100. * batch_idx / len(self.data_loader.train_loader),
            loss_val,
        ))
        # log to tensorboard
        self.summary_writer.add_scalar(
            tag='Loss/train',
            scalar_value=loss_val,
            global_step=self.current_iteration,
        )

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
                self._log_train_iter(batch_idx=batch_idx, loss_val=loss_val)

            self.current_iteration += 1

            if self.debug:
                break

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

                if self.debug:
                    break

        val_loss /= len(self.data_loader.val_loader.dataset)

        self.logger.info(f'\nValidation set: Average loss: {val_loss:.4f}n')

        # log to tensorboard
        self.summary_writer.add_scalar(
            tag='Loss/validation',
            scalar_value=val_loss,
            global_step=self.current_epoch,
        )

    def finalize(self):
        """
        Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
        :return:
        """
        pass

    @property
    def num_train_samples(self):
        return len(self.data_loader.train_loader.dataset)

    @property
    def num_val_samples(self):
        return len(self.data_loader.val_loader.dataset)
