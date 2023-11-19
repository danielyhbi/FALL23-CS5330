# Daniel Bi - CS5330 CV
# Homework 5, part 1
# 11/20/2023

# import statements
import sys
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ===================================================
# initialize run environment as non-function calls
learning_rate = 0.01
momentum = 0.5
log_interval = 10
n_epochs = 3
mnist_global_mean = 0.1307
mnist_global_stdev = 0.3081
random_seed = 1
batch_size_train = 64
batch_size_test = 1000
epochs = 5
train_losses = []
train_counter = []
test_losses = []
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)
# ===================================================

# class definitions
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        # Feature Extraction Stage
        self.feature_maps_stack = nn.Sequential(
            # A convolution layer with 10 5x5 filters
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5),
            # A max pooling layer with a 2x2 window and a ReLU function applied
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            # A convolution layer with 20 5x5 filters
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5),
            # A dropout layer with a 0.5 dropout rate (50%)
            nn.Dropout2d(p=0.5),
            # A max pooling layer with a 2x2 window and a ReLU function applied
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU()
        )
        
        # Classification Stage
        self.classify_stack = nn.Sequential(
            # A flattening operation 
            nn.Flatten(),
            # followed by a fully connected Linear layer with 50 nodes 
            nn.Linear(320, 50),
            # and a ReLU function on the output
            nn.ReLU(),
            # A final fully connected Linear layer with 10 nodes
            nn.Linear(50, 10)
        )
        
    def forward(self, x):
        features = self.feature_maps_stack(x)
        logits = self.classify_stack(features)
        
        # and the log_softmax function applied to the output
        return nn.functional.log_softmax(logits, dim=1)

# useful functions with a comment for each function
def train_network_helper(train_dataloader, model, epoch, optimizer, device):
    model.train()
    for batch, (data, target) in enumerate(train_dataloader):
        # send the data to hardware device
        data, target = data.to(device), target.to(device)
        # reset zero gradients and forward pass in model
        optimizer.zero_grad()
        output = model(data)
        # compute loss function
        loss = nn.functional.nll_loss(output, target)
        # backward pass to computer gradients
        loss.backward()
        # update model with optimizer
        optimizer.step()

        if batch % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch * len(data), len(train_dataloader.dataset),
                100. * batch / len(train_dataloader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
                (batch*64) + ((epoch-1)*len(train_dataloader.dataset)))
            
            torch.save(model.state_dict(), 'model.pth')
            torch.save(optimizer.state_dict(), 'optimizer.pth')
    return

# Obtain the training and testing data
def get_dataloaders(mean, stdev) -> ():
    
    transformer = transforms.Compose([transforms.ToTensor(), transforms.Normalize((mean,), (stdev,))])

    # Download training data from open datasets.
    training_data = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transformer,
    )

    # Download test data from open datasets.
    test_data = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transformer,
    )
    
    train_dataloader = DataLoader(training_data, batch_size=batch_size_train, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size_test, shuffle=True)

    for X, y in test_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break
    
    return train_dataloader, test_dataloader

# Get cpu, gpu or mps device for training. 
def get_device() -> str:
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")
    
    return device

# delegate method to train and evaluate the model
def train_model(train_dataloader, test_dataloader, model, optimizer, device):
    
    validate_model(model, test_dataloader, device)
    for t in range(1, n_epochs + 1):
        print(f"Epoch {t+1}\n-------------------------------")
        train_network_helper(train_dataloader, model, t, optimizer, device)
        validate_model(model, test_dataloader, device)
    print("Done!")
    
    return

# helpter method to evaluate the model
def validate_model(model, test_dataloader, device):
    # enable the testing mode
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_dataloader:
            # send the data to hardware device
            data, target = data.to(device), target.to(device)
            output = model(data)
            # Compute the negative log likelihood loss
            test_loss += nn.functional.nll_loss(output, target, size_average=False).item()
            # Get the index of the maximum log-probability
            pred = output.data.max(1, keepdim=True)[1]
            # Count the number of correct predictions
            correct += pred.eq(target.data.view_as(pred)).sum()
    
    # Compute average loss        
    test_loss /= len(test_dataloader.dataset)
    test_losses.append(test_loss)
    
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_dataloader.dataset),
        100. * correct / len(test_dataloader.dataset)))
    return

# helper method to plot out the train loss
def plot_train_performace(test_counter):
    
    fig = plt.figure()
    plt.plot(train_counter, train_losses, color='orange')
    plt.scatter(test_counter, test_losses, color='green')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.show()
    
    return

# helper mehtod to plot out incoming data
def plot_data(train_dataloader):
    
    examples = enumerate(train_dataloader)
    batch_idx, (example_data, example_targets) = next(examples)

    fig = plt.figure()
    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Ground Truth: {}".format(example_targets[i]))
        plt.xticks([])
        plt.yticks([])
    plt.show()
    
    return

# main function (yes, it needs a comment too)
def main(argv):
    
    # obtain dataloaders
    train_dataloader, test_dataloader = get_dataloaders(mnist_global_mean, mnist_global_stdev)
    
    test_counter = [i*len(train_dataloader.dataset) for i in range(n_epochs + 1)]
    
    # obtain model
    device = get_device()
    model = NeuralNetwork().to(device)
    print(model)
    
    # prepare optimizer for training
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    train_model(train_dataloader, test_dataloader, model, optimizer, device)
    
    plot_train_performace(test_counter)
    
    return

if __name__ == "__main__":
    main(sys.argv)