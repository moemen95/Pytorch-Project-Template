"""
The Base Agent class, where all other agents inherit from, that contains definitions for all the necessary functions
"""
import logging

import gin
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from graphs.optimizers import sgd
from utils.devices import configure_device
import utils.dirs as module_dirs


class BaseAgent:
    """
    This base class will contain the base functions to be overloaded by any agent you will implement.
    """

    def __init__(self):
        self.logger = logging.getLogger("Agent")
        self.agent_name = 'Base'

    def _get_state_dict(self):
        return {
            # TODO dump config here
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
        return gin.query_parameter('%debug')


@gin.configurable
class BaseTrainAgent(BaseAgent):
    def __init__(
            self,
            max_epoch,
            log_interval=10,
            checkpoint_path=None,
    ):
        super().__init__()

        self.agent_name = 'BaseTrainAgent'
        self.max_epoch = max_epoch
        self.log_interval = log_interval

        self._init_counters()
        self._init_model()
        self._init_optimizer()
        self._init_data_loader()
        self._init_device()
        self._init_tboard_logging()

        self.load_checkpoint(checkpoint_path)

    def _init_counters(self):
        self.current_epoch = 0
        self.current_iteration = 0

    def _init_model(self):
        raise NotImplementedError()

    def _init_optimizer(self):
        self.optimizer = sgd(params=self.model.parameters())

    def _init_data_loader(self):
        raise NotImplementedError()

    def _init_device(self):
        self.device = configure_device()
        self.model = self.model.to(self.device)

    def _init_tboard_logging(self):
        self.summary_writer = SummaryWriter(
            log_dir=module_dirs.get_current_tboard_dir(),
            comment=self.agent_name,
        )

    def _get_state_dict(self):
        state_dict = super()._get_state_dict()

        state_dict.update({
            'epoch': self.current_epoch,
            'iteration': self.current_iteration,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'torch_random_state': torch.get_rng_state(),
            'numpy_random_state': np.random.get_state(),
        })

        return state_dict

    def _load_state_dict(self, state_dict):
        self.current_epoch = state_dict['epoch'] + 1
        self.current_iteration = state_dict['iteration']
        self.model.load_state_dict(state_dict['model_state_dict'])
        self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])

        torch.set_rng_state(state_dict['torch_random_state'])
        np.random.set_state(state_dict['numpy_random_state'])

    @property
    def checkpoints_dir(self):
        return module_dirs.get_current_checkpoints_dir()

    @property
    def experiments_dir(self):
        # checkpoint_dir is of structure .../exp_name/datetime_str/
        # and we will look for the checkpoint file under a different timestamp
        return self.checkpoints_dir.parent

    def load_checkpoint(self, file_name=None):
        """
        Load model from the latest checkpoint - if not specified start from scratch.
        :param file_name: name of the checkpoint file
        :return:
        """
        if file_name is not None and len(file_name) > 0:
            checkpoint_path = self.experiments_dir / file_name

            self.logger.info(f'Loading checkpoint "{checkpoint_path}"')

            checkpoint = torch.load(checkpoint_path)
            self._load_state_dict(checkpoint)

            self.logger.info(
                f"""Checkpoint loaded successfully from '{checkpoint_path}' at (epoch {checkpoint['epoch']}) 
                at (iteration {checkpoint['iteration']})\n""")

    def save_checkpoint(self, is_best=False):
        """
        Checkpoint saver
        :param is_best: boolean flag to indicate whether current checkpoint's accuracy is the best so far
        :return:
        """
        checkpoint_path = self.checkpoints_dir / f'epoch_{self.current_epoch}.pth'
        torch.save(self._get_state_dict(), checkpoint_path)
        if is_best:
            best_checkpoint_path = self.checkpoints_dir / 'best.pth'
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
        for epoch in range(self.current_epoch, self.max_epoch):
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

            if batch_idx % self.log_interval == 0:
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
        with torch.no_grad():
            for data, target in self.data_loader.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                # sum up batch loss
                val_loss += self.loss_fn(output, target, size_average=False).item()

                if self.debug:
                    break

        val_loss /= len(self.data_loader.val_loader.dataset)

        self.logger.info(f'\nValidation set: Average loss: {val_loss:.4f}')

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
