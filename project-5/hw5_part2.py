# Daniel Bi - CS5330 CV
# Homework 5, part 2
# 11/20/2023

import cv2
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
device = hw5.get_device()
model = hw5.NeuralNetwork().to(device)
model.load_state_dict(torch.load("model.pth"))
# ===================================================

# part A, helper to visualize the first layer of the network
def visualize_network():
    
    first_layer_conv1 = model.feature_maps_stack[0]
    # transfer the weight info from cuda to cpu
    first_layer_conv1_weight = first_layer_conv1.weight.cpu()

    print("Filter Weights Shape:", first_layer_conv1_weight.shape)
    
    # Visualize the filters using Pyplot
    fig = plt.figure(figsize=(10, 8))
    for i in range(10):
        plt.subplot(3, 4, i+1)
        plt.imshow(first_layer_conv1_weight[i, 0].detach().numpy(), cmap='BuPu')
        plt.title(f'Filter {i + 1}')
        plt.xticks([]), plt.yticks([])
    plt.show()
    
    return

# visualize the filter effect with openCV
def visualize_filter_effect(train_dataloader):
    
    for data, _ in train_dataloader:
        first_example = data[0]
        break
    
    first_layer_conv1 = model.feature_maps_stack[0]
    # transfer the weight info from cuda to cpu
    first_layer_conv1_weight = first_layer_conv1.weight.cpu()

    # Apply the filters using filter2D
    with torch.no_grad():
        filtered_images = []
        for i in range(10):
            # Extract a single filter
            filter_i = first_layer_conv1_weight[i, 0].squeeze().numpy()

            # Apply the filter using filter2D
            filtered_image = cv2.filter2D(first_example.squeeze().numpy(), -1, filter_i)
            
            # Convert back to the range [0, 255]
            filtered_image = ((filtered_image - filtered_image.min()) / (filtered_image.max() - filtered_image.min()) * 255).astype('uint8')

            filtered_images.append(filtered_image)
    
    # Visualize the filtered images using Pyplot
    fig = plt.figure(figsize=(10, 8))
    for i in range(10):
        # plot the filter with conv1 weights
        plt.subplot(5, 4, 2*(i)+1)
        plt.imshow(first_layer_conv1_weight[i, 0].detach().numpy(), cmap='BuPu')
        plt.title(f'Filter {i + 1}')
        plt.xticks([]), plt.yticks([])
        
        # plot the filter2D
        plt.subplot(5, 4, 2*(i+1))
        plt.imshow(filtered_images[i], cmap='gray')
        plt.title(f'Filter {i + 1}')
        plt.xticks([]), plt.yticks([])
        
    plt.show()
    
    return

# main function (yes, it needs a comment too)
def main():
    
    _, train_dataloader = hw5.get_dataloaders(mean=mnist_global_mean, stdev=mnist_global_stdev)
    
    visualize_network()
    
    visualize_filter_effect(train_dataloader)
    
    return

if __name__ == "__main__":
    main()