# PyTorch-Project-Template

Implement your PyTorch projects the smart way.

A scalable template for PyTorch projects, with examples in Image Segmentation, Object classification, GANs and Reinforcement Learning.

This is a joint work between [Hager Rady](https://github.com/hagerrady13/) and [Mo'men AbdelRazek](https://github.com/moemen95)

### Why this template?

We are proposing a baseline for any PyTorch project to give you a quick start, where you will get the time to focus in your model's implementation and we will handle the rest. The novelty of this approach lies in:
- Providing a scalable project structure, with a template file for each.
- Introducing the usage a config file that handle all the hyper-parameters related to a certain problem.
- Embedding examples from various problems inside the template.
- Tutorials to get you started.

### Tutorials:
We are providing a series of tutorials to get your started

* [Getting Started from Scratch Tutorial](https://github.com/moemen95/PyTorch-Project-Template/blob/master/tutorials/template_tutorial.md) where we provide a guide on the main steps to get started on your project.
* [Mnist tutorial](https://github.com/moemen95/PyTorch-Project-Template/blob/master/tutorials/mnist_tutorial.md): Here we take an already implemented NN model on Mnist and adapt it to our template structure.

### Contribution:
* We are welcoming any contribution from the community that may add value to the template. 
* We aim that this template can be a central place for different examples of the well-known PyTorch Deep learning models. 
* We are also welcoming any proposed changes about the design pattern used in this project.

### Template Class Diagram:

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

1) We started by DCGAN, adding its custom configs into the json file. DCGAN has both generator and discriminator model so it doesn't have a single model file.
2) Then, we added CondenseNet, where it was necessary to create a custom blocks folder inside the models folder to include the definitions for custom layers within the model.
3) After that, we added the DQN project, where all the classes related to the environment have been added to the utils. We also added the action selection and model optimization into the training agent.

This is to ensure that our proposed project structure is compatible with different problems and can handle all the variations related to any of them.

### Requirements:
```
Pytorch: 0.4.0
torchvision: 0.2.1
tensorboardX: 1.2
gym: 0.10.5
tqdm: 4.23.3
easydict: 1.7
```

### To-Do:

We are planning to add more examples into our template to include various categories of problems. Next we are going to include the following:

* MobilenetV2
* visual-interaction-networks-pytorch
* variational-Autoencoder-pytorch

### License:
This project is licensed under MIT License - see the LICENSE file for details
