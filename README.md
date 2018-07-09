# PyTorch-Project-Template

Implement your PyTorch projects the smart way.

A scalable template for PyTorch projects, with examples in Image Segmentation, Object classification, GANs and Reinforcement Learning.

This template is created by [Hager Rady](https://github.com/hagerrady13/) & [Mo'men AbdelRazek](https://github.com/moemen95)

### Template Class Diagram:

### Why this template?

### Repo Structure:
```
├── agents
|  └── dcgan.py
|  └── condensenet.py
|  └── mnist.py
|  └── dqn.py
|  └── example.py
|
├── configs
|  └── dcgan_exp_0.py
|  └── condensenet_exp_0.py
|  └── mnist_exp_0.py
|  └── dqn_exp_0.py
|  └── example_exp_0.py
|
├── data
|
├── datasets
|  └── cifar10.py
|  └── celebA.py
|  └── mnist.py
|  └── example.py
|
├── experiments
|
├── graphs
|  └── models
|  |  └── custome_layers
|  |  |  └── denseblock.py
|  |  |  └── layers.py
|  |  |
|  |  └── dcgan.py
|  |  └── condensenet.py
|  |  └── mnist.py
|  |  └── dqn.py
|  |  └── example.py
|  |
|  └── losses
|  |  └── loss.py
|
├── pretrained_weights
|
├── tutorials
|
├── utils
|  └── assets
|
├── main.py
└── run.sh
```

### Referenced Repos:
1. [FCN8s](https://github.com/hagerrady13/FCN8s-Pytorch): A model for semantic **Segmentation**, trained on Pascal Voc
2. [DCGAN](https://github.com/hagerrady13/DCGAN-Pytorch): Deep Convolutional **Generative Adverserial Networks**, run on CelebA dataset.
3. [CondenseNet](https://github.com/hagerrady13/CondenseNet-Pytorch): A model for Image **Classification**, trained on Cifar10 dataset
4. [DQN](https://github.com/hagerrady13/DQN-Pytorch): Deep Q Network model, a **Reinforcement Learning** example, tested on CartPole-V0

### Repos Migration Summary:

### To-Do:

### Requirements:
```
Pytorch: 0.4.0
torchvision: 0.2.1
tensorboardX: 1.2
gym: 0.10.5
tqdm: 4.23.3
easydict: 1.7
```

### Contribution:
We are welcoming any contribution from the community that may add value to the template.
### License:
This project is licensed under MIT License - see the LICENSE file for details
