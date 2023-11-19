# Daniel Bi - CS5330 CV
# Homework 5, part 2
# 11/20/2023

import sys
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchviz import make_dot

import hw5_part1 as hw5

# ===================================================
# initialize run environment as non-function calls
learning_rate = 0.01
momentum = 0.5
log_interval = 10
n_epochs = 20
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
device = hw5.get_device()
# ===================================================

# greek data set transform
class GreekTransform:
    def __init__(self):
        pass

    def __call__(self, x):
        x = transforms.functional.rgb_to_grayscale( x )
        x = transforms.functional.affine( x, 0, (0,0), 36/128, 0 )
        x = transforms.functional.center_crop( x, (28, 28) )
        return transforms.functional.invert( x )

# freeze the pre-trained weights from the loaded model
def modify_model(model):
    # freeze the model weights
    for param in model.parameters():
        param.requires_grad = False
        
    # replace the last layer with custom
    model.classify_stack[-1] = nn.Linear(in_features=50, out_features=3)

    print(model)
    return

# plots to visualize the model for output
def visualize_model(model, greek_train_dataloader):
    
    examples = enumerate(greek_train_dataloader)
    batch_idx, (example_data, example_targets) = next(examples)

    sample_pred = model(example_data.to(device)).cpu()
    make_dot(sample_pred, params=dict(model.named_parameters()))
    
    return

# load the greek letter training data
def get_dataloaders(mean, stdev) -> ():
    
    training_set_path = "/Users/danielbi/git-repo/FALL23-CS5330/playground/pyTorch/data/greek_train"

    # DataLoader for the Greek data set
    greek_train_dataloader = torch.utils.data.DataLoader(
        datasets.ImageFolder(training_set_path,
                            transform = transforms.Compose([transforms.ToTensor(),
                                                            GreekTransform(),
                                                            transforms.Normalize((0.1307,),(0.3081,))
                                                            ])
                            ),
        batch_size = 5,
        shuffle = True )

    greek_test_dataloader = torch.utils.data.DataLoader(
        datasets.ImageFolder(training_set_path,
                            transform = transforms.Compose([transforms.ToTensor(),
                                                            GreekTransform(),
                                                            transforms.Normalize((0.1307,),(0.3081,))
                                                            ])
                            ),
        batch_size = 5,
        shuffle = True )
    
    print("Train data summary:")
    for X, y in greek_train_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break
    
    print("Test data summary:")
    for X, y in greek_train_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break
    
    return greek_train_dataloader, greek_test_dataloader

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
            
            torch.save(model.state_dict(), 'model3.pth')
            torch.save(optimizer.state_dict(), 'optimizer3.pth')
    return

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

# actual delegate method to facilitate training and validation
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

# plot the train loss of the model
def plot_train_performace(test_counter):
    
    fig = plt.figure()
    plt.plot(train_counter, train_losses, color='orange')
    plt.scatter(test_counter, test_losses, color='green')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.show()
    
    return

# get my own handwriting for the validation
def get_eval_dataloader():
    validate_set_path = "/Users/danielbi/git-repo/FALL23-CS5330/playground/pyTorch/data/greek_test"

    greek_valid_dataloader = torch.utils.data.DataLoader(
        datasets.ImageFolder(validate_set_path,
                            transform = transforms.Compose([transforms.Resize((28, 28)),
                                                            transforms.Grayscale(),
                                                            transforms.ToTensor(),
                                                            transforms.Normalize((0.1307,),(0.3081,))
                                                            ])
                            ),
        batch_size = 6,
        shuffle = True )
    
    print("\nThe eval data stats:")
    for X, y in greek_valid_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break
    
    return greek_valid_dataloader

# helper method to validate the model
def eval(model, data_loader):
    
    examples = enumerate(data_loader)
    batch_idx, (example_data, example_targets) = next(examples)
    
    with torch.no_grad():
        example_data, example_targets = example_data.to(device), example_targets.to(device)
        output = model(example_data)
        probabilities = F.softmax(output, dim=1)
        predicted = output.data.max(1, keepdim=True)[1]
        
        top_p, top_class = probabilities.topk(1, dim = 1)
        
        for i in range(len(example_data)):
            print(f"Example {i + 1} - Correct Label: {example_targets[i]}, Predicted Label: {top_class[i].item()}, Probabilities: {top_p[i].item():.2f}")

    return

# main function (yes, it needs a comment too)
def main():
    
    # get the model from part 1
    device = hw5.get_device()
    model = hw5.NeuralNetwork().to(device)
    model.load_state_dict(torch.load("model.pth"))
    
    modify_model(model)
    model = hw5.NeuralNetwork().to(device)
    
    train_dataloader, test_dataloader = get_dataloaders(mnist_global_mean, mnist_global_stdev)
    
    test_counter = [i*len(train_dataloader.dataset) for i in range(n_epochs + 1)]
    
    # prepare optimizer for training
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    train_model(train_dataloader, test_dataloader, model, optimizer, device)
    
    eval_dataloader = get_eval_dataloader()
    
    eval(model, eval_dataloader)
    
    plot_train_performace(test_counter)
    
    return

if __name__ == "__main__":
    main()