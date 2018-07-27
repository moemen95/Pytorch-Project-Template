"""
Cifar10 Dataloader implementation, used in CondenseNet
"""
import logging
import numpy as np

import torch
import torchvision.transforms as v_transforms
import torchvision.utils as v_utils
import torchvision.datasets as v_datasets

from torch.utils.data import DataLoader, TensorDataset, Dataset


class Cifar10DataLoader:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("Cifar10DataLoader")
        if config.data_mode == "numpy_train":

            self.logger.info("Loading DATA.....")
            normalize = v_transforms.Normalize(mean=[0.4914, 0.4824, 0.4467],
                                             std=[0.2471, 0.2435, 0.2616])

            train_set = v_datasets.CIFAR10('./data', train=True, download=True,
                                         transform=v_transforms.Compose([
                                             v_transforms.RandomCrop(32, padding=4),
                                             v_transforms.RandomHorizontalFlip(),
                                             v_transforms.ToTensor(),
                                             normalize,
                                         ]))
            valid_set = v_datasets.CIFAR10('./data', train=False,
                                       transform=v_transforms.Compose([
                                           v_transforms.ToTensor(),
                                           normalize,
                                       ]))

            self.train_loader = DataLoader(train_set, batch_size=self.config.batch_size, shuffle=True)
            self.valid_loader = DataLoader(valid_set, batch_size=self.config.batch_size, shuffle=False)

            self.train_iterations = len(self.train_loader)
            self.valid_iterations = len(self.valid_loader)

        elif config.data_mode == "numpy_test":
            test_data = torch.from_numpy(np.load(config.data_folder + config.x_test)).float()
            test_labels = torch.from_numpy(np.load(config.data_folder + config.y_test)).long()

            self.len_test_data = test_data.size()[0]

            self.test_iterations = (self.len_test_data + self.config.batch_size - 1) // self.config.batch_size

            self.logger.info("""
                Some Statistics about the testing data
                test_data shape: {}, type: {}
                test_labels shape: {}, type: {}
                test_iterations: {}
            """.format(test_data.size(), test_data.type(), test_labels.size(), test_labels.type(),
                       self.test_iterations))

            test = TensorDataset(test_data, test_labels)

            self.test_loader = DataLoader(test, batch_size=config.batch_size, shuffle=False)

        elif config.data_mode == "random":
            train_data = torch.randn(self.config.batch_size, self.config.input_channels, self.config.img_size, self.config.img_size)
            train_labels = torch.ones(self.config.batch_size).long()
            valid_data = train_data
            valid_labels = train_labels
            self.len_train_data = train_data.size()[0]
            self.len_valid_data = valid_data.size()[0]

            self.train_iterations = (self.len_train_data + self.config.batch_size - 1) // self.config.batch_size
            self.valid_iterations = (self.len_valid_data + self.config.batch_size - 1) // self.config.batch_size

            train = TensorDataset(train_data, train_labels)
            valid = TensorDataset(valid_data, valid_labels)

            self.train_loader = DataLoader(train, batch_size=config.batch_size, shuffle=True)
            self.valid_loader = DataLoader(valid, batch_size=config.batch_size, shuffle=False)

        else:
            raise Exception("Please specify in the json a specified mode in data_mode")

    def finalize(self):
        pass
