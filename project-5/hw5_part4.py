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
from datetime import datetime

# ===================================================
# initialize run environment as non-function calls
# so i don't pass 100000 variables in to a function
learning_rate = 0.01
momentum = 0.5
log_interval = 100
# n_epochs = 3
mnist_global_mean = 0.1307
mnist_global_stdev = 0.3081
random_seed = 1
batch_size_train = 64
batch_size_test = 1000
# epochs = 5
train_losses = []
train_counter = []
test_losses = []
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)
# ===================================================

# class definitions
class NeuralNetwork(nn.Module):
    # channelSize is on the increment
    def __init__(self, convoCount, channelSize, dropoutRate, pixelSize):
        super(NeuralNetwork, self).__init__()

        # Feature Extraction Stage
        self.feature_maps_stack = nn.Sequential()
        startChannel = 1
        
        for count in range(convoCount):
            self.feature_maps_stack.append(nn.Conv2d(in_channels=startChannel,
                                                         out_channels=channelSize * (count + 1),
                                                         kernel_size=5))
            pixelSize -= 4
            
            if count >= 1:
                self.feature_maps_stack.append(nn.Dropout2d(p=dropoutRate))

            if count >= 1 and count < 3:
                self.feature_maps_stack.append(nn.MaxPool2d(kernel_size=2))
                # print("Before divide by 2")
                # print(pixelSize)
                pixelSize = int(pixelSize / 2)
                # print("After divide by 2")
                # print(pixelSize)
                
            self.feature_maps_stack.append(nn.ReLU())
            
            # reset env
            startChannel = channelSize * (count + 1)
        
        totalNode = pixelSize * pixelSize * startChannel
        
        # Classification Stage
        self.classify_stack = nn.Sequential(
            # A flattening operation 
            nn.Flatten(),
            # followed by a fully connected Linear layer with 50 nodes 
            nn.Linear(totalNode, 50),
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
def train_network_helper(train_dataloader, model, loss_fn, optimizer, device):
    size = len(train_dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return

# Obtain the training and testing data
def get_dataloaders(mean, stdev) -> ():
    
    transformer = transforms.Compose([transforms.ToTensor()])

    # Download training data from open datasets.
    training_data = datasets.FashionMNIST(
        root="./data",
        train=True,
        download=True,
        transform=transformer,
    )

    # Download test data from open datasets.
    test_data = datasets.FashionMNIST(
        root="./data",
        train=False,
        download=True,
        transform=transformer,
    )
    
    train_dataloader = DataLoader(training_data, batch_size=batch_size_train, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size_test, shuffle=True)
        
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
    
    return device

# helper method to faciliate the training and validation process
def train_model(train_dataloader, test_dataloader, model, loss_fn, optimizer, device, test_result, epochs):
    
    for t in range(epochs):
        print(f"Epoch {t+1}")
        # test_result.append(f"Epoch {t+1}\n-------------------------------")
        train_network_helper(train_dataloader, model, loss_fn, optimizer, device)
        validate_model(model, loss_fn, test_dataloader, device, test_result, t)
    print("Done!")
    
    return

# helper method to validate the model
def validate_model(model, loss_fn, test_dataloader, device, test_result, epoCount):
    # enable the testing mode
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_dataloader:
            # send the data to hardware device
            data, target = data.to(device), target.to(device)
            pred = model(data)
            
            test_loss += loss_fn(pred, target).item()
            correct += (pred.argmax(1) == target).type(torch.float).sum().item()
    
    # Compute average loss        
    test_loss /= len(test_dataloader)
    test_losses.append(test_loss)
    
    result = 'Test set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_dataloader.dataset),
        100. * correct / len(test_dataloader.dataset))
    
    print(result)
    
    log = '{},{:.4f},{:.0f}'.format(epoCount+1,
                                    test_loss, 
                                    100. * correct / len(test_dataloader.dataset))
    
    test_result.append(log)
    
    return

# main model helper
def make_model(convoCount, channelSize, dropoutRate, pixelSize):
    # obtain dataloaders
    train_dataloader, test_dataloader = get_dataloaders(mnist_global_mean, mnist_global_stdev)
    
    test_result = []
    
    # obtain model
    device = get_device()
    model = NeuralNetwork(convoCount, channelSize, dropoutRate, pixelSize).to(device)
    print(model)
    # test_result.append("MODEL INFO:")
    # test_result.append(str(model))
    
    # prepare optimizer for training
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    train_model(train_dataloader, test_dataloader, model, loss_fn, optimizer, device, test_result, 20)
    
    print(test_result[-1])
    
    return test_result

# main function (yes, it needs a comment too)
def main():
    
    log = []
    count = 1
    pixelSize = 28
    convolutionList = [1, 2, 3]
    channelIncrement = [10, 20, 30]
    dropoutRateList = [0.5, 0.7]
    
    log.append("test,convoCount,channelSize,dropoutRate,epochs,avgLoss,accuracy")
    
    for convoCount in convolutionList:
        for channelSize in channelIncrement:
            for dropoutRate in dropoutRateList:
                #for epochs in epochsList:
                    print("Starting test {}...".format(count))
                    
                    message = 'Final result for convoCount: {}, channelSize: {}, dropoutRate: {}'.format(
                        convoCount, channelSize, dropoutRate)
                    print(message)
                    
                    run_info = '{},{},{},{},'.format(count ,convoCount, channelSize, dropoutRate)
                    
                    startTime = datetime.now()
                    
                    result = make_model(convoCount, channelSize, dropoutRate, pixelSize)
                    
                    endTime = datetime.now()
                    
                    timeDiff = (endTime - startTime).total_seconds()
                    
                    for item in result:
                        log.append(run_info + item)
                        
                    print('Time used: {}s'.format(timeDiff))
                    print('===============\n')
                    
                    count += 1
    
    with open('log.txt', 'w') as file:
        for item in log:
            file.write(item + "\n")
    
    return

if __name__ == "__main__":
    main()