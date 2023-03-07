import torch.nn as nn
import torch.nn.functional as F

# Define the model
class NeuralNet(nn.Module):
    """A class representing a deep learning model for image classification.
    
    Args:
        in_channels (int): The number of channels in the input data (1 for grayscale, 3 for RGB).
        out_channels (int): The number of predicted classes.
        kernel (int): The size of the convolutional kernel.
    """
    def __init__(self, in_channels, out_channels, kernel):
        super(NeuralNet, self).__init__()
        # Define 3 convolutional layers
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=kernel, stride=1, padding=1) 
        self.conv2 = nn.Conv2d(32, 64, kernel_size=kernel, stride=1, padding=1) 
        self.conv3 = nn.Conv2d(64, 128, kernel_size=kernel, stride=1, padding=1) 
        # Define 2 fully connected layers 
        self.fc1 = nn.Linear(8 * 8 * 128, 128)
        self.fc2 = nn.Linear(128, out_channels)
        
    def forward(self, x):
        """The forward pass of the model.
        
        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, in_channels, height, width).
        
        Returns:
            torch.Tensor: The output tensor of shape (batch_size, out_channels).
        """
        # Apply ReLU activation and max pooling for each conv layer
        x = F.relu(F.max_pool2d(self.conv1(x), 2)) 
        x = F.relu(F.max_pool2d(self.conv2(x), 2)) 
        x = F.relu(F.max_pool2d(self.conv3(x), 2))

        # Flatten the tensor based on batch size
        x = x.view(x.size(0), -1) 
        x = F.relu(self.fc1(x)) 
        x = self.fc2(x)
        return x