import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from DataGenerator import DataGeneration
from NeuralNet import NeuralNet


class Evaluation:
    """
    A class to evaluate the performance of a trained Neural Network model on test data
    """

    def __init__(self, model_dir, data_dir):
        """
        Initializes the Evaluation class

        Args:
            model_dir (str): Path to the directory containing the trained model's files
            data_dir (str): Path to the directory containing the dataset
        """
        self.model_dir = model_dir
        self.data_dir = data_dir

    def evaluate(self):
        """
        Evaluates the performance of the trained model on test data

        Returns:
            None
        """
        # Set the device to use (either CPU or GPU)
        device = ("cuda" if torch.cuda.is_available() else "cpu")

        # Load the trained NeuralNet model
        Net = NeuralNet(1, 3, 3).to(device)
        Net.load_state_dict(torch.load(os.path.join(
            self.model_dir, "model_shape.pth"), map_location=device))

        # Load the test dataset
        test_dataset = DataGeneration(self.data_dir, 64, False)

        # Define dataloader for the test dataset
        test_loader = DataLoader(
            test_dataset, batch_size=1, shuffle=False)

        # Set the model in evaluation mode
        with torch.no_grad():
            Net.eval()

            # Initialize the loss and correct count
            loss = 0
            correct = 0
            y_pred = []
            y_true = []

            # Iterate over the test dataloader
            for data, target in tqdm(test_loader):
                data, target = data.to(device), target.to(device)
                output = Net(data)
                loss += F.cross_entropy(
                    output, target, reduction='sum').item()

                # Get the predicted class index
                _, pred = torch.max(output.data, 1)
                correct += (pred == target).sum().item()

                y_pred.extend(pred.cpu().numpy())
                y_true.extend(target.numpy())

            # Compute average loss and accuracy
            loss /= len(test_loader.dataset)
            accuracy = 100. * correct / len(test_loader.dataset)

            # Print the average loss and accuracy
            print('Average Val Loss: {:.4f}, Val Accuracy: {}/{} ({:.3f}%)\n'.format(
                loss, correct, len(test_loader.dataset), accuracy))

            # Save the predicted and true class labels

            
            if not os.path.exists('./results/'):
                os.makedirs('./results/')
            np.savetxt('./results/y_pred.txt', y_pred, fmt='%d')
            np.savetxt('./results/y_true.txt', y_true, fmt='%d')

            # Plot the confusion matrix
            self.plot_confusion_matrix(y_true, y_pred)

    def plot_confusion_matrix(self, y_true, y_pred):
        """
        Plots the confusion matrix

        Args:
            y_true (numpy array): True class labels
            y_pred (numpy array): Predicted class labels

        Returns:
            None
        """
        # Calculate confusion matrix
        cf_matrix = confusion_matrix(y_true, y_pred)

        # Define class names
        class_names = ('circle', 'square', 'rectangle')

        # Create a dataframe from confusion matrix
        dataframe = pd.DataFrame(cf_matrix, index=class_names, columns=class_names)

        # Plot the confusion matrix using Seaborn heatmap
        sns.heatmap(dataframe, annot=True, cbar=None, cmap='YlGnBu', fmt='d')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.ylabel('True Class')
        plt.xlabel('Predicted Class')

        # Save the confusion matrix plot to file
        plt.savefig('./results/cm_matrix.jpg', bbox_inches='tight')


if __name__ == "__main__":

    evaluation = Evaluation('./model_dir/', './data/shapes/')
    evaluation.evaluate()