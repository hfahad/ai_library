# Import required libraries and modules
import os
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from DataGenerator import DataGeneration  # Import custom data generator
from NeuralNet import NeuralNet  # Import custom neural network model

from torch.utils.tensorboard import SummaryWriter


writer = SummaryWriter('./runs/shape')

class Trainer:
    def __init__(self, batch_size=64, nb_epochs=20, model_dir='./model_dir', in_channels=1, out_channels=10, kernel=3):
        """
        Constructor for the Trainer class.

        Parameters:
            batch_size (int): Batch size for training and validation data.
            nb_epochs (int): Number of epochs for training.
            model_dir (str): Directory path to save the trained model.
            in_channels (int): Number of channels in input images (1 for grayscale, 3 for RGB).
            out_channels (int): Number of output classes.
            kernel (int): Convolutional kernel size.
        """
        self.batch_size = batch_size
        self.nb_epochs = nb_epochs
        self.model_dir = model_dir
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel = kernel

    def train(self):
        # Load data
        train_dataset = DataGeneration('./data/shapes/', 64, True)
        trainData, valData = train_test_split(train_dataset, random_state=10, test_size=0.25)

        # Create data loaders for training and validation data
        train_loader = DataLoader(trainData, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(valData, batch_size=self.batch_size, shuffle=False)

        # Use GPU if available, otherwise use CPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create a neural network model object
        net = NeuralNet(in_channels=self.in_channels, out_channels=self.out_channels, kernel=self.kernel)
        net.to(device)
        net.train()

        # Define optimizer and loss function
        optimizer = optim.Adam(params=net.parameters(), lr=0.002)
        criterion = nn.CrossEntropyLoss()

        # Train the model for specified number of epochs
        for n in range(self.nb_epochs):
            correct = 0
            running_loss = 0
            print(f'Epoch: {n+1}/{self.nb_epochs}')

            # Train the model on batches of data
            for (data, target) in tqdm(train_loader):
                data, target = data.to(device), target.to(device)

                optimizer.zero_grad()
                output = net(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                # Calculate accuracy and loss for current batch
                _, pred = torch.max(output.data, 1)
                correct += (pred == target).sum().item()
                running_loss += loss.item()

            # Print average training loss and accuracy for current epoch
            print('\nAverage training Loss: {:.4f}, training Accuracy: {}/{} ({:.3f}%)\n'.format(
                loss, correct, len(train_loader.dataset), 100. * correct / len(train_loader.dataset)))
            
            # Tensorboard defining
            writer.add_scalar('training loss',
                          running_loss / len(train_loader.dataset),
                          n)

            writer.add_scalar('Accuracy',
                          correct / len(train_loader.dataset),
                          n)

            with torch.no_grad():
                net.eval()
                loss = 0
                correct = 0

                # Evaluate the model on validation data
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)

                    output = net(data)
                    loss += F.cross_entropy(output, target, reduction='sum').item()

                    _, pred = torch.max(output.data, 1)
                    correct += (pred == target).sum().item()

                loss /= len(test_loader.dataset)

                print('Average Val Loss: {:.4f}, Val Accuracy: {}/{} ({:.3f}%)\n'.format(
                    loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        path = self.model_dir + '/model_shape.pth'
        torch.save(net.state_dict(), path)


if __name__ == "__main__":
    parser = ArgumentParser()

    # Add argument for batch size
    parser.add_argument("--batch_size", type=int,
                        dest="batch_size", default=64,
                        help="batch_size")

    # Add argument for number of epochs
    parser.add_argument("--epochs", type=int,
                        dest="nb_epochs", default=15,
                        help="number of iterations")

    # Add argument for data folder
    parser.add_argument("--model_dir", type=str,
                        dest="model_dir", default='./model_dir',
                        help="model save folder")

    # Add argument for data folder
    parser.add_argument("--in_channels", type=int,
                        dest="in_channels", default=1,
                        help="1 for GRAY and 3 for RGB")

    # Add argument for data folder
    parser.add_argument("--out_channels", type=int,
                        dest="out_channels", default=3,
                        help="Total numbers of classes")

    # Add argument for data folder
    parser.add_argument("--kernel", type=int,
                        dest="kernel", default=3,
                        help="convolutional kernel size")

    args = parser.parse_args()
    trainer = Trainer(**vars(args))
    trainer.train()