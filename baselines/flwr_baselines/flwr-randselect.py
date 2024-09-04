from collections import OrderedDict
from typing import Optional, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10
import sys
import random
from numpy.random import default_rng
import flwr as fl
from flwr.common import Metrics
import argparse


# Check if CUDA is available and set the device accordingly
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
    
print(
    f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}"
)

CLASSES = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

NUM_CLIENTS = 10


BATCH_SIZE = 32

parser = argparse.ArgumentParser()
parser.add_argument("--strategy", type=str, default='dirichlet')
parser.add_argument('--alpha', type=float, default=0.4)
parser.add_argument('--fraction_fit', type=float, default=0.8)
args = parser.parse_args()
print(f"Strategy: {args.strategy}, Alpha: {args.alpha}, Fraction Fit: {args.fraction_fit}")
def load_datasets(strategy: str, alpha:float) -> Tuple[List[DataLoader], List[DataLoader], DataLoader]:

    # Download and transform CIFAR-10 (train and test)
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    trainset = CIFAR10("./dataset", train=True, download=True, transform=transform)
    testset = CIFAR10("./dataset", train=False, download=True, transform=transform)

    if strategy == 'equal':
        # Split training set into 10 partitions to simulate the individual dataset
        partition_size = len(trainset) // NUM_CLIENTS
        lengths = [partition_size] * NUM_CLIENTS
        datasets = random_split(trainset, lengths, torch.Generator().manual_seed(42))

        num_classes = len(CLASSES)  # Adjust this based on your dataset
        # Initialize class counts for each client
        client_class_counts = [{i: 0 for i in range(num_classes)} for _ in range(NUM_CLIENTS)]

        # Assuming trainset.targets exists and contains all labels
        all_labels = np.array(trainset.targets)

        # Calculate start and end indices for each client's data partition
        for client_idx, dataset in enumerate(datasets):
            indices = dataset.indices  # Get the indices of samples for this client's subset
            labels = all_labels[indices]  # Directly access the labels for this subset

            # Count occurrences of each class in this subset
            for class_idx in range(num_classes):
                client_class_counts[client_idx][class_idx] = np.sum(labels == class_idx)

        # Display the number of samples and class distributions for each client
        for client_idx, dataset in enumerate(datasets):
            print(f"Client {client_idx} has {len(dataset)} samples.")
        for client_idx, dataset in enumerate(datasets):
            print(f"Client {client_idx} class distribution: {client_class_counts[client_idx]}")


    #strategy and components added 11/19
    elif strategy == 'dirichlet':
        # Define alpha and class_proportions for the Dirichlet distribution
        alpha = args.alpha
        class_proportions = None  # Set to None or a dictionary of class proportions

        class_indices = {i: np.where(np.array(trainset.targets) == i)[0] for i in range(len(CLASSES))}
        client_data_indices = [[] for _ in range(NUM_CLIENTS)]
        client_class_counts = [{i: 0 for i in range(len(CLASSES))} for _ in range(NUM_CLIENTS)]

        for class_idx, indices in class_indices.items():
            proportions = np.random.dirichlet(alpha=np.repeat(alpha, NUM_CLIENTS))
            if class_proportions and class_idx in class_proportions:
                proportions *= class_proportions[class_idx]
            proportions /= proportions.sum()

            np.random.shuffle(indices)
            allocated = 0
            for client_idx, proportion in enumerate(proportions):
                num_samples = int(proportion * len(indices))
                selected_indices = indices[allocated: allocated + num_samples]
                client_data_indices[client_idx].extend(selected_indices)
                client_class_counts[client_idx][class_idx] += len(selected_indices)
                allocated += num_samples

        # Display the number of samples for each client
        for client_idx, indices in enumerate(client_data_indices):
            print(f"Client {client_idx} has {len(indices)} samples.")
        for client_idx, class_count in enumerate(client_class_counts):
            print(f"Client {client_idx} class distribution: {class_count}")

        datasets = [torch.utils.data.Subset(trainset, indices) for indices in client_data_indices]
        trainloaders = []
        valloaders = []
        for client_indices in client_data_indices:
            client_dataset = torch.utils.data.Subset(trainset, client_indices)

            # Splitting client dataset into train and validation sets
            len_val = len(client_dataset) // 10  # 10% validation set
            len_train = len(client_dataset) - len_val
            ds_train, ds_val = random_split(client_dataset, [len_train, len_val], torch.Generator().manual_seed(42))

            # Creating DataLoaders for train and validation sets
            trainloaders.append(DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True))
            valloaders.append(DataLoader(ds_val, batch_size=BATCH_SIZE))
    if strategy == 'dirichlet-fixed':
        alpha = args.alpha
        rng_seed = 42
        rng = default_rng(rng_seed) 

        class_indices = {i: np.where(np.array(trainset.targets) == i)[0] for i in range(len(CLASSES))}
        client_data_indices = [[] for _ in range(NUM_CLIENTS)]
        client_class_counts = [{i: 0 for i in range(len(CLASSES))} for _ in range(NUM_CLIENTS)]

        for class_idx, indices in class_indices.items():
            # Use the seeded RNG for Dirichlet distribution
            proportions = rng.dirichlet(alpha=np.repeat(alpha, NUM_CLIENTS))
            proportions /= proportions.sum()  # Ensure proportions sum to 1

            # Shuffle class indices using the seeded RNG
            rng.shuffle(indices)

            allocated = 0
            for client_idx, proportion in enumerate(proportions):
                num_samples = int(proportion * len(indices))
                selected_indices = indices[allocated: allocated + num_samples]
                client_data_indices[client_idx].extend(selected_indices)
                client_class_counts[client_idx][class_idx] += len(selected_indices)
                allocated += num_samples

        # Filter out clients with zero samples to prevent DataLoader issues
        valid_client_indices = [indices for indices in client_data_indices if len(indices) > 0]
        datasets = [torch.utils.data.Subset(trainset, indices) for indices in valid_client_indices]
        trainloaders = []
        valloaders = []

        for client_indices in client_data_indices:
            client_dataset = torch.utils.data.Subset(trainset, client_indices)

            # Splitting client dataset into train and validation sets
            len_val = len(client_dataset) // 10  # 10% validation set
            len_train = len(client_dataset) - len_val
            ds_train, ds_val = random_split(client_dataset, [len_train, len_val], torch.Generator().manual_seed(42))

            # Creating DataLoaders for train and validation sets
            trainloaders.append(DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True))
            valloaders.append(DataLoader(ds_val, batch_size=BATCH_SIZE))

        # Display the number of samples for each client
        for client_idx, indices in enumerate(client_data_indices):
            print(f"Client {client_idx} has {len(indices)} samples.")
        for client_idx, class_count in enumerate(client_class_counts):
            print(f"Client {client_idx} class distribution: {class_count}")
    else:
        raise ValueError(f"Invalid strategy: {strategy}")

     #Split each partition into train/val and create DataLoader
    trainloaders = []
    valloaders = []
    for ds in datasets:
        len_val = len(ds) // 10  # 10 % validation set
        len_train = len(ds) - len_val
        lengths = [len_train, len_val]
        ds_train, ds_val = random_split(ds, lengths, torch.Generator().manual_seed(42))
        trainloaders.append(DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True))
        valloaders.append(DataLoader(ds_val, batch_size=BATCH_SIZE))
  # testloader = DataLoader(testset, batch_size=BATCH_SIZE)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE)
    return trainloaders, valloaders, testloader


trainloaders, valloaders, testloader = load_datasets(args.strategy,args.alpha)
print("Dataset loaded with Strategy: ", args.strategy,args.alpha)  


images, labels = next(iter(trainloaders[0]))

# Reshape and convert images to a NumPy array
# matplotlib requires images with the shape (height, width, 3)
images = images.permute(0, 2, 3, 1).numpy()
# Denormalize
images = images / 2 + 0.5

print(labels)
print(type(labels))

#New CNN

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten(),
            nn.Linear(256*4*4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10))

    def forward(self, xb):
        return self.network(xb)

"""Layer 0: Distance Before Adjustment: 0.26384660601615906, After Adjustment: 0.26384660601615906
Layer 2: Distance Before Adjustment: 2.1425867080688477, After Adjustment: 0.21425867080688477
Layer 5: Distance Before Adjustment: 0.15106679499149323, After Adjustment: 0.30213358998298645
Layer 7: Distance Before Adjustment: 0.11768963187932968, After Adjustment: 0.14122755825519562
Layer 10: Distance Before Adjustment: 8.11756706237793, After Adjustment: 8.11756706237793
Layer 12: Distance Before Adjustment: 17.596616744995117, After Adjustment: 8.798308372497559
Layer 16: Distance Before Adjustment: 0.8256460428237915, After Adjustment: 0.8256460428237915
"""

#Shiyue Net
#interpretation of NN sent
#11/10
# class SimpleConvNet(nn.Module):
#     def __init__(self, in_channels, num_classes, dropout_rate=0):
#         super(SimpleConvNet, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, 32, 3, stride=1, padding=2)
#         self.conv2 = nn.Conv2d(32, 32, 3, stride=1, padding=2)
#         self.conv3 = nn.Conv2d(32, 32, 3, stride=1, padding=2)
#         self.conv4 = nn.Conv2d(32, 32, 3, stride=1, padding=2)

#         self.gn_relu = nn.Sequential(
#             nn.GroupNorm(32, 32, affine=True),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#         )


#         example_input = torch.zeros(1, in_channels, 32, 32)
#         example_output = self.conv4(self.gn_relu(self.conv3(self.gn_relu(self.conv2(
#             self.gn_relu(self.conv1(self.gn_relu(example_input))))))))
#         num_features = example_output.numel()

#         self.dropout = nn.Dropout(dropout_rate)
#         self.fc = nn.Linear(num_features, num_classes)

#     def forward(self, x):
#         x = self.gn_relu(self.conv1(x))
#         x = self.gn_relu(self.conv2(x))
#         x = self.gn_relu(self.conv3(x))
#         x = self.gn_relu(self.conv4(x))
#         x = x.view(-1, self.num_flat_features(x))
#         x = self.fc(self.dropout(x))
#         return x

# class Net(nn.Module):
#     def __init__(self) -> None:
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 16 * 5 * 5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         x = F.softmax(x,dim = 1)
#         return x

#standard Network

# class Net(nn.Module):
#     def __init__(self) -> None:
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 2)  # Kernel size is 2
#         self.pool = nn.MaxPool2d(2, 2)
#         self.fc1 = nn.Linear(6 * 15 * 15, 50)  # Adjusted based on the new tensor size
#         self.fc2 = nn.Linear(50, 10)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.pool(F.relu(self.conv1(x)))
#         x = x.view(-1, 6 * 15 * 15)  # Adjusted based on the new kernel size
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         x = F.softmax(x, dim=1)
#         return x

# simplified network

# class Net(nn.Module):
#     def __init__(self) -> None:
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.conv3 = nn.Conv2d(16, 32, 5)  # New convolutional layer
#         # Since we're not pooling after the third convolutional layer, the size remains (32x1x1).
#         self.fc1 = nn.Linear(32 * 1 * 1, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = F.relu(self.conv3(x))  # Forward pass through the new convolutional layer without pooling
#         x = x.view(-1, 32 * 1 * 1)  # Updated view/flattening shape
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
# #complex network

# # #ResNet Implementation

# from torch import Tensor
# from typing import Type

# class BasicBlock(nn.Module):
#     def __init__(
#         self,
#         in_channels: int,
#         out_channels: int,
#         stride: int = 1,
#         expansion: int = 1,
#         downsample: nn.Module = None
#     ) -> None:
#         super(BasicBlock, self).__init__()
#         # Multiplicative factor for the subsequent conv2d layer's output channels.
#         # It is 1 for ResNet18 and ResNet34.
#         self.expansion = expansion
#         self.downsample = downsample
#         self.conv1 = nn.Conv2d(
#             in_channels,
#             out_channels,
#             kernel_size=3,
#             stride=stride,
#             padding=1,
#             bias=False
#         )
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(
#             out_channels,
#             out_channels*self.expansion,
#             kernel_size=3,
#             padding=1,
#             bias=False
#         )
#         self.bn2 = nn.BatchNorm2d(out_channels*self.expansion)
#     def forward(self, x: Tensor) -> Tensor:
#         identity = x
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#         out = self.conv2(out)
#         out = self.bn2(out)
#         if self.downsample is not None:
#             identity = self.downsample(x)
#         out += identity
#         out = self.relu(out)
#         return  out

# class Net(nn.Module):
#     def __init__(
#         self,
#         img_channels: int,
#         num_layers: int,
#         block: Type[BasicBlock],
#         num_classes: int  = 1000
#     ) -> None:
#         super(Net, self).__init__()
#         if num_layers == 18:
#             # The following `layers` list defines the number of `BasicBlock`
#             # to use to build the network and how many basic blocks to stack
#             # together.
#             layers = [2, 2, 2, 2]
#             self.expansion = 1


#         self.in_channels = 64
#         # All ResNets (18 to 152) contain a Conv2d => BN => ReLU for the first
#         # three layers. Here, kernel size is 7.
#         self.conv1 = nn.Conv2d(
#             in_channels=img_channels,
#             out_channels=self.in_channels,
#             kernel_size=7,
#             stride=2,
#             padding=3,
#             bias=False
#         )
#         self.bn1 = nn.BatchNorm2d(self.in_channels)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.layer1 = self._make_layer(block, 64, layers[0])
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(512*self.expansion, num_classes)
#     def _make_layer(
#         self,
#         block: Type[BasicBlock],
#         out_channels: int,
#         blocks: int,
#         stride: int = 1
#     ) -> nn.Sequential:
#         downsample = None
#         if stride != 1:
#             """
#             This should pass from `layer2` to `layer4` or
#             when building ResNets50 and above. Section 3.3 of the paper
#             Deep Residual Learning for Image Recognition
#             (https://arxiv.org/pdf/1512.03385v1.pdf).
#             """
#             downsample = nn.Sequential(
#                 nn.Conv2d(
#                     self.in_channels,
#                     out_channels*self.expansion,
#                     kernel_size=1,
#                     stride=stride,
#                     bias=False
#                 ),
#                 nn.BatchNorm2d(out_channels * self.expansion),
#             )
#         layers = []
#         layers.append(
#             block(
#                 self.in_channels, out_channels, stride, self.expansion, downsample
#             )
#         )
#         self.in_channels = out_channels * self.expansion
#         for i in range(1, blocks):
#             layers.append(block(
#                 self.in_channels,
#                 out_channels,
#                 expansion=self.expansion
#             ))
#         return nn.Sequential(*layers)
#     def forward(self, x: Tensor) -> Tensor:
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         # The spatial dimension of the final layer's feature
#         # map should be (7, 7) for all ResNets.
#         # print('Dimensions of the last convolutional feature map: ', x.shape)
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.fc(x)
#         return x

def count_model_layers(model):
    layer_count = 0
    for _ in model.network:
        layer_count += 1
    return layer_count

def get_learnable_layer_indices(model):
    learnable_layers = []
    for idx, layer in enumerate(model.network):
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            learnable_layers.append(idx)
            print(f"Learnable Layer_{idx}: {type(layer).__name__}")
    return learnable_layers

def get_layer_indices_by_type(model):
    """Categorizes learnable layers into CNN and linear layers."""
    cnn_layer_indices = []
    linear_layer_indices = []
    for idx, layer in enumerate(model.network):
        if isinstance(layer, nn.Conv2d):
            cnn_layer_indices.append(idx)
        elif isinstance(layer, nn.Linear):
            linear_layer_indices.append(idx)
    return cnn_layer_indices, linear_layer_indices

model = Net()
learnable_layer_indices = get_learnable_layer_indices(model)
total_layers = count_model_layers(model)
print(f"Total number of layers in the model: {total_layers}")
learnable_layer_indices = get_learnable_layer_indices(model)
print(learnable_layer_indices)

def train(net, trainloader, epochs: int, verbose=False):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())
    net.train()
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # Metrics
            epoch_loss += loss
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total
        if verbose:
            print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")
        return epoch_loss, epoch_acc #return loss 3/31

def test(net, testloader):
    """Evaluate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy

# Code to Train CNN

trainloader = trainloaders[0]
valloader = valloaders[0]
net = Net().to(DEVICE)

# Code to Train CNN

def get_parameters(net) -> List[np.ndarray]:
    parameters = [val.cpu().numpy() for _, val in net.state_dict().items()]
    # print(f"Parameters size: {get_params_size(parameters)} bytes")
    return parameters


def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


# Code to train ResNet
# def get_parameters(net) -> List[np.ndarray]:
#     return [val.cpu().numpy() for name, val in net.state_dict().items() if 'num_batches_tracked' not in name]

# def set_parameters(net, parameters: List[np.ndarray]):
#     keys = [k for k in net.state_dict().keys() if 'num_batches_tracked' not in k]
#     params_dict = zip(keys, parameters)
#     state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
#     net.load_state_dict(state_dict, strict=True)

def get_params_size(parameters):
    total_size = 0
    for p in parameters:
        total_size += sys.getsizeof(p)
        if isinstance(p, np.ndarray):
            total_size += p.nbytes
    return total_size

from flwr.common.parameter import ndarrays_to_parameters as nd2p
from flwr.common.parameter import parameters_to_ndarrays as p2nd
import time
from flwr.server.client_manager import ClientManager
# import logging
# import multiprocessing

class FlowerClient(fl.client.NumPyClient):

      def __init__(self, cid, net, trainloader, valloader):
          self.cid = cid
          self.net = net
          self.trainloader = trainloader
          self.valloader = valloader

      def get_parameters(self, config):
          return get_parameters(self.net)

      def fit(self, parameters, config):
          set_parameters(self.net, parameters)
          start_time = time.time()
          train_loss, train_accuracy = train(self.net, self.trainloader, epochs=1)  # Capture returned metrics
          end_time = time.time()
          total_time = end_time - start_time
          adj_total_time = "%.3f" % total_time
          parameters = get_parameters(self.net)

          return parameters, len(self.trainloader), {"loss": train_loss, "accuracy": train_accuracy} #add loss and accuracy to return 3/31

      def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader)
        print(f"Client ID: {self.cid}, Loss: {loss}, Accuracy: {accuracy}")
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}

#code to train cnn
def client_fn(cid: str) -> FlowerClient:
    """Create a Flower client representing a single organization."""

    # Load model
    net = Net().to(DEVICE)

    # Load data (CIFAR-10)
    # Note: each client gets a different trainloader/valloader, so each client
    # will train and evaluate on their own unique data
    trainloader = trainloaders[int(cid)]
    valloader = valloaders[int(cid)]
    return FlowerClient(cid, net, trainloader, valloader)

#code to train resnet
# def client_fn(cid: str) -> FlowerClient:
#     """Create a Flower client representing a single organization."""


#     net = Net(3,18,BasicBlock).to(DEVICE)
#     trainloader = trainloaders[int(cid)]
#     valloader = valloaders[int(cid)]
#     return FlowerClient(cid, net, trainloader, valloader)

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}

#function to remove old files (possibly for colab only*)

import os
import glob

file_key = "client*.txt"
for del_file in glob.glob(file_key):
  os.remove(del_file)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

"""The only thing left to do is to tell the strategy to call this function whenever it receives evaluation metric dictionaries from the clients:"""

from numpy.core.multiarray import asanyarray
from copy import deepcopy
#Custom Strategy

"""Federated Averaging (FedAvg) [McMahan et al., 2016] strategy.

Paper: arxiv.org/abs/1602.05629
"""


from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg, _compute_distances
from flwr.server.strategy import Strategy

WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """
Setting `min_available_clients` lower than `min_fit_clients` or
`min_evaluate_clients` can cause the server to fail when there are too few clients
connected to the server. `min_available_clients` must be set to a value larger
than or equal to the values of `min_fit_clients` and `min_evaluate_clients`.
"""

class CustStrat(Strategy):
    """Configurable FedAvg strategy implementation."""

    def __init__(
        self,
        *,
        # training_client_ids: str,
        # validation_client_ids: str,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
    ) -> None:

        super().__init__()

        if (
            min_fit_clients > min_available_clients
            or min_evaluate_clients > min_available_clients
        ):
            log(WARNING, WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW)
        # self.training_client_ids=training_client_ids
        # self.validation_client_ids=validation_client_ids
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.evaluate_fn = evaluate_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn


    def __repr__(self) -> str:
        rep = f"FedAvg(accept_failures={self.accept_failures})"
        return rep

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return the sample size and the required number of available clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        initial_parameters = self.initial_parameters
        self.initial_parameters = None  # Don't keep initial parameters in memory
        return initial_parameters

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        if self.evaluate_fn is None:
            # No evaluation function provided
            return None
        parameters_ndarrays = parameters_to_ndarrays(parameters)
        eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})
        if eval_res is None:
            return None
        loss, metrics = eval_res
        return loss, metrics

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]


    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
        ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of evaluation."""
        # Do not configure federated evaluation if fraction eval is 0.
        if self.fraction_evaluate == 0.0:
            return []

        # Parameters and config
        config = {}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(server_round)
        evaluate_ins = EvaluateIns(parameters, config)

        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        return [(client, evaluate_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Convert results
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results))

        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated



    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Aggregate loss
        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")

        return loss_aggregated, metrics_aggregated

from numpy.core.multiarray import asanyarray
from copy import deepcopy



#Random Selection
strategy = CustStrat(
    fraction_fit=args.fraction_fit,
    fraction_evaluate=1.0,
    min_fit_clients=1,
    min_evaluate_clients=1,
    min_available_clients=1,
    evaluate_metrics_aggregation_fn=weighted_average,  # <-- pass the metric aggregation function
)


client_resources = None
if DEVICE.type == "cuda":
    client_resources = {"num_gpus": 1}

# Start simulation
fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=50),
    strategy=strategy,
    client_resources=client_resources,
)

file_key = "client*.txt"
file_name = "collect_logs"
input_file_name = file_name + '.txt'
with open(input_file_name, "w") as outfile:
  for file in glob.glob(file_key):
    with open(file, 'r') as infile:
      outfile.write(infile.read())
      outfile.write("\n")

