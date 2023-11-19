# Daniel Bi - CS5330 CV
# Homework 5, part 1F
# 11/20/2023

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os

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
device = hw5.get_device()
model = hw5.NeuralNetwork().to(device)
model.load_state_dict(torch.load("model.pth"))
# ===================================================

# custom dataset that loads my handwriting into the dataloader
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.images = self._load_images()

    def _load_images(self):
        images = []
        for cls in self.classes:
            if not cls.startswith('.'):
                class_path = os.path.join(self.root_dir, cls)
                for img_name in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_name)
                    #images.append((img_path, self.class_to_idx[cls]))
                    label_tensor = torch.tensor(int(img_name[4]))
                    images.append((img_path, label_tensor))
        return images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        img = Image.open(img_path).convert('L')

        if self.transform:
            img = self.transform(img)

        return img, label

# helper function to load custom data
def load_custom_data():
    # load custom data
    transformer = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((mnist_global_mean,), (mnist_global_stdev,))
        ])

    new_dataset = CustomDataset(
        root_dir='/Users/danielbi/git-repo/FALL23-CS5330/playground/pyTorch/data/SelfWriting', 
        transform=transformer
        )
    
    data_loader = DataLoader(new_dataset, batch_size=10, shuffle=False)
    
    return data_loader

# helper method to classify the additional handwriting images
def eval(data_loader):
    
    model.eval()
    
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
    
    data_loader = load_custom_data()
    
    eval(data_loader)
    
    for X, y in data_loader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break
    
    return

if __name__ == "__main__":
    main()