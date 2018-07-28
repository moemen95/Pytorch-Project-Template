# Mnist Tutorial
This is a detailed tutorial on how to adapt your PyTorch code into our project structure.

We will walk through a PyTorch [basic model on Mnist](https://github.com/pytorch/examples/blob/master/mnist/main.py) and transform it into our template format.

### 1. The config file:
This is the core contribution of our template. We duplicate '/configs/exmaple_exp_0.json' and rename it to [mnist_exp_0.json](https://github.com/moemen95/PyTorch-Project-Template/blob/master/configs/mnist_exp_0.json).
We rename the agent and dataloader to the ones for Mnist. As we go, we will add and modify the configurations keys and values whenever needed.


### 2. Main model:

We start by the model definition, defined here in our [reference](https://github.com/pytorch/examples/blob/master/mnist/main.py)

```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
```
We will move this model definition into the folder '/graphs/models' with the name mnist.py. The class is renamed into 'Mnist'.
A minor change will be adding the weight initializer at the end of the model constructor, after importing.
```python
from ..weights_initializer import weights_init

class Mnist(nn.Module):
    def __init__(self):
        super(Mnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        self.apply(weights_init)
```
Now, the modified model can be found [here](https://github.com/moemen95/PyTorch-Project-Template/blob/master/graphs/models/mnist.py).

### 3. Loss:
The example is using nll_loss as a function called during training and test times. We usually add a class for our loss into the folder 'graphs/losses'.
Since we don't define our own loss function, we can use the same one from Pytorch directly.
```python
self.loss = nn.NLLLoss()
```

### 4. DataLoader:
We add a new file named [mnist.py](https://github.com/moemen95/PyTorch-Project-Template/blob/master/datasets/mnist.py) in the folder '/datasets'. The class is renamed into 'MnistDataLoader'.
Below is the main part concerned with data loading, so we add it into the init of MnistDataLoader
```python
class MnistDataLoader:
    def __init__(self, config):
        """
        :param config:
        """
        self.config = config
        # Notice the usage of the data mode here
        if config.data_mode == "download":
            self.train_loader = torch.utils.data.DataLoader(
                datasets.MNIST('../data', train=True, download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))
                               ])),
                batch_size=self.config.batch_size, shuffle=True, num_workers=self.config.data_loader_workers, pin_memory=self.config.pin_memory)
            self.test_loader = torch.utils.data.DataLoader(
                datasets.MNIST('../data', train=False, transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])),
                batch_size=self.config.test_batch_size, shuffle=True, num_workers=self.config.data_loader_workers, pin_memory=self.config.pin_memory)
```
Since we added a new mode named 'download' as we are downloading data inside the loader and not saved inside '/data', we need to edit some fields in the configurations as follows:
```json
  "data_loader": "MnistDataLoader",
  "data_loader_workers": 2,
  "pin_memory": true,
  "async_loading": true,

  "data_mode": "download",
  "data_folder": ""
```
And we will also add ```"test_batch_size": 1000 ``` inside the [config file]((https://github.com/moemen95/PyTorch-Project-Template/blob/master/configs/mnist_exp_0.json))

All the config parameters used can be accessed inside the code from within ``` self.config ```

### 5. Agent
This is where all the action take places. We create a new file named [mnist.py](https://github.com/moemen95/PyTorch-Project-Template/blob/master/agents/mnist.py) in the folder '/agents' and use the given [example.py](https://github.com/moemen95/PyTorch-Project-Template/blob/master/agents/example.py) for our reference.
#### The agent init:
Before you write the agent, you need to edit the following fields in the [config file](https://github.com/moemen95/PyTorch-Project-Template/blob/master/configs/mnist_exp_0.json)
```json
  "learning_rate": 0.01,
  "momentum": 0.5,
```
Then move the to the main agent,
```Python
# Import the model we defined before
from graphs.models.mnist import Mnist
# Import the dataloader as we defined before
from datasets.mnist import MnistDataLoader

# We inhert our agent from the base agent to implement all the needed functions in the base 
class MnistAgent(BaseAgent):

    def __init__(self, config):
        super().__init__(config)

        # create instance from the model
        self.model = Mnist()

        # Create instance from the dataloder given the config dictionary
        self.data_loader = MnistDataLoader(config=config)

        # define loss
        self.loss = nn.NLLLoss()
        
        # define optimizers, given the right parameters
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.config.learning_rate, momentum=self.config.momentum)

        # initialize counters
        self.current_epoch = 0
        self.current_iteration = 0
        self.best_metric = 0

        # Call Load checkpoint function to load weights from the latest checkpoint, if not found start from scratch.
        self.load_checkpoint(self.config.checkpoint_file)
        # Summary Writer
        self.summary_writer = None
```
#### Train function:
In our [reference](https://github.com/pytorch/examples/blob/master/mnist/main.py) , we have this code inside the main function that should move to the function 'train'
```python
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)
```
We don't need to pass all these parameters, so it will be changed into
```python
    for epoch in range(1, self.config.max_epoch + 1):
        self.train_one_epoch()
        self.validate()
```
- Also, add ``` self.current_epoch += 1 ```
Don't forget to update the max_epoch field in the config file with that value given in the main args.

#### Train One epoch function:
In our [reference](https://github.com/pytorch/examples/blob/master/mnist/main.py), we have the code responsible for model training inside the function 'train'. This code will be moved to the function train_one_epoch inside our agent as follows, with slight changes:

```python
def train_one_epoch(self):
    """
    One epoch of training
    :return:
    """
    # Notice that model has changed to 'self.model'
    self.model.train()
    # Also, train_loader has changed into self.data_laoder.train_loader    
    for batch_idx, (data, target) in enumerate(self.data_loader.train_loader):
        data, target = data.to(self.device), target.to(self.device)
        self.optimizer.zero_grad()
        output = self.model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        self.optimizer.step()
        if batch_idx % self.config.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                self.current_epoch, batch_idx * len(data), len(self.data_loader.train_loader.dataset),
                       100. * batch_idx / len(self.data_loader.train_loader), loss.item()))
        # We add an update to the current iteration
        self.current_iteration += 1
```
- ``` "log_interval": 10 ``` field should be added into the [config file](https://github.com/moemen95/PyTorch-Project-Template/blob/master/configs/mnist_exp_0.json).

#### Validate function:

In our [reference](https://github.com/pytorch/examples/blob/master/mnist/main.py), we have the code responsible for model training inside the function 'test'. This code will be moved to the function validate_one_epoch inside our agent as follows, with slight changes:

```python
def validate(self):
    """
    One cycle of model validation
    :return:
    """
    # Notice that model has changed to 'self.model'
    self.model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        # Also, test_loader has changed into self.data_laoder.test_loader    
        for data, target in self.data_loader.test_loader:
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(self.data_loader.test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(self.data_loader.test_loader.dataset),
        100. * correct / len(self.data_loader.test_loader.dataset)))
```

### 5. Model Verification:

In the [reference example](https://github.com/pytorch/examples/blob/master/mnist/main.py)'s main function, Go through the arguments parameters and make sure they are included inside the [config file](https://github.com/moemen95/PyTorch-Project-Template/blob/master/configs/mnist_exp_0.json), with the correct values.
e.g. seed value

- To run your code, change the config file name inside run.sh to be ``` mnist_exp_0.json ```
- On the terminal, run ``` sh run.sh ```
- Verify the results relative to the original example

## Summary:
Model, agent and dataloader are the main building blocks in the template. The provided examples can be used as a start to migrate any Pytorch model into our template structure.