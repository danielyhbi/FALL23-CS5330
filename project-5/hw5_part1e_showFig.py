# Daniel Bi - CS5330 CV
# Homework 5, part 1e
# 11/20/2023

# import statements
import sys
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt

import hw5_part1 as hw5

# ===================================================
# initialize run environment as non-function calls
mnist_global_mean = 0.1307
mnist_global_stdev = 0.3081
# load the device
device = hw5.get_device()
model = hw5.NeuralNetwork().to(device)
model.load_state_dict(torch.load("model.pth"))
# ===================================================

# load data but only get the first 10 per homework instruction
# returns a test_dataloader with only 10 data
def load_data():
  
  transformer = transforms.Compose([transforms.ToTensor(), transforms.Normalize((mnist_global_mean,), (mnist_global_stdev,))])
  
  # Download test data from open datasets.
  test_data = datasets.MNIST(
      root="./data",
      train=False,
      download=True,
      transform=transformer,
  )
  
  subset_index = list(range(10))
  
  subset_test_dataloader = Subset(test_data, subset_index)
  
  return DataLoader(subset_test_dataloader, batch_size=10, shuffle=False)

# helper method to evaluate the computed model
def eval(subset_test_dataloader):
  
  model.eval()
  
  for X, y in subset_test_dataloader:
      print(f"Shape of X [N, C, H, W]: {X.shape}")
      print(f"Shape of y: {y.shape} {y.dtype}")
      break

  examples = enumerate(subset_test_dataloader)
  batch_idx, (example_data, example_targets) = next(examples)

  with torch.no_grad():
      example_data, example_targets = example_data.to(device), example_targets.to(device)
      output = model(example_data)
      probabilities = F.softmax(output, dim=1)
      predicted = output.data.max(1, keepdim=True)[1]
      
      top_p, top_class = probabilities.topk(1, dim = 1)

      for i in range(len(example_data)):
          print(f"Example {i + 1} - Correct Label: {example_targets[i]}, Predicted Label: {top_class[i].item()}, Probabilities: {top_p[i].item():.2f}")

  # convert the example data back to local
  example_data_local = [tensor.cpu() for tensor in example_data]
  
  return example_data_local, output

# plto out the result with
def plot_result(example_data_local, output):
  fig = plt.figure()

  for i in range(9):
    plt.subplot(3,3,i+1)
    plt.tight_layout()
    plt.imshow(example_data_local[i][0], cmap='gray', interpolation='none')
    plt.title("Prediction: {}".format(
      output.data.max(1, keepdim=True)[1][i].item()))
    plt.xticks([])
    plt.yticks([])
    
  plt.show()
  
  return

# main function (yes, it needs a comment too)
def main():
  
  # get the subset test data
  subset_test_dataloader = load_data()
  
  # evaluate model
  example_data_local, output = eval(subset_test_dataloader)
  
  # plot result
  plot_result(example_data_local, output)
  
  return

if __name__ == "__main__":
    main()