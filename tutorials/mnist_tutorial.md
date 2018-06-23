* Mnist Tutorial
** Adapting Mnist Example into the template

We will walk through a Pytorch basic example on Mnist and transform it into our project structure.

0. The config file:
This is the core contribution of our template. We duplicate '/configs/exmaple_exp_0.json' and rename it to [mnist_exp_0.json](https://github.com/moemen95/PyTorch-Project-Template/blob/master/configs/mnist_exp_0.json).
Rename the values inside the config file. As we go, we will add and modify the configurations keys and values whenever needed.


1. Main model:

We start by the model definition, defined here in the example

```
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
```
We will move this model definition into the folder '/graphs/models' with the name mnist.py. The class is renamed into 'Mnist'.
A slight change will be adding the weight initializer at the end of the model constructor, after importing.
```
  from ..weights_initializer import weights_init

  self.apply(weights_init)
```
You can find it [here](https://github.com/moemen95/PyTorch-Project-Template/blob/master/graphs/models/mnist.py).


2. Loss:
The example is using nll_loss as a function called during training and test times. We usually add a class for our loss into the folder 'graphs/losses'.
Since we don't define our own loss function, we can use the nll loss function from Pytorch directly.


3. DataLoader:
We duplicate the example file in we have in the folder '/datasets' and rename it into [mnist.py](https://github.com/moemen95/PyTorch-Project-Template/blob/master/datasets/mnist.py). The class is renamed into 'MnistDataLoader'.
These is the main part concerned with data loading.
```
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
```
In /datasets/mnist.py, we added a new mode named 'download' as we are downloading data inside the loader and not saved inside '/data'
In the configurations, we need to edit the following fields from
```
  "data_loader": "ExampleDataLoader",
  "data_loader_workers": 1,
  "pin_memory": true,
  "async_loading": true,

  "data_mode": "imgs",
  "data_folder": "./data/example"
```
to:
```
  "data_loader": "MnistDataLoader",
  "data_loader_workers": 2,
  "pin_memory": true,
  "async_loading": true,

  "data_mode": "download",
  "data_foldr": ""
```
And we will add ```"test_batch_size": 1000 ``` inside the [config file]((https://github.com/moemen95/PyTorch-Project-Template/blob/master/configs/mnist_exp_0.json))

The parameters used are now passed from withing self.config


4. Agent:
We duplicate the example file in we have in the folder '/agents' and rename it into mnist.py.
*** In the constructor,
- We define our model, imported from '/graphs/models/mnist.py'. Don't forget to the import statement at the beginning of the file
```
from graphs.models.mnist import Mnist

self.model = Mnist(self.config)
```
- We define our dataloader, imported from '/datasets/MnistDataLoader.py'. Don't forget to the import statement at the beginning of the file

```
from datasets.mnist import MnistDataLoader

self.data_loader = MnistDataLoader(config=config)
```
- We define the optimizer,
```
  self.optimizer = optim.SGD(self.model.parameters(), lr=self.config.learning_rate, momentum=self.config.momentum)

```
And edit these fields in the [config file](https://github.com/moemen95/PyTorch-Project-Template/blob/master/configs/mnist_exp_0.json)
```
  "learning_rate": 0.01,
  "momentum": 0.5,
```
There is nothing else to be modified in the constructor.
*** Train function:
In our reference example, we have this code inside the main function that should move to the function 'train'
```
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)
```
We don't need to pass all these parameters, so it will be changed into
```
    for epoch in range(1, self.config.max_epoch + 1):
        self.train_one_epoch()
        self.validate()
```
- add ``` self.current_epoch += 1 ```
Don't forget to update the max_epoch field in the config file with that value given in the main args.

*** Train One epoch function:
This function will include all what is inside the function 'train'

```
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))
```

We will copy this into 'train_one_epoch' with no parameters to be passed
It is only summarized in naming changes.
- ```model``` changed into ``` self.model ```
- ``` "log_interval": 10 ``` field should be added into the [config file](https://github.com/moemen95/PyTorch-Project-Template/blob/master/configs/mnist_exp_0.json).
- ``` train_loader ``` is changed into ``` self.data_laoder.train_loader ```
- add ``` self.current_iteration += 1 ```
*** Validate function:

This function will include all what is inside the function 'test'

```
def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
```

We will copy this into 'validate' with no parameters to be passed
It is only summarized in naming changes.
- ```model``` changed into ``` self.model ```
- ``` test_loader ``` is changed into ``` self.data_laoder.test_loader ```

5. In the reference example main function, Go through the arguments parameters and make sure they are included inside the [config file](https://github.com/moemen95/PyTorch-Project-Template/blob/master/configs/mnist_exp_0.json), with the correct values.
e.g. seed value
