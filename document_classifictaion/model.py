# Importing Required Libraries
# Torch is a Python library for building and training deep learning models.
# nn is a module within PyTorch that provides functionalities to define neural network layers and architectures.
# F provides functions for commonly used operations in neural networks (e.g., pooling, activation).
# torchsummary is used to generate a summary of the model.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


# Residual Convolutional Block
# This block uses two convolutional layers and a shortcut connection (residual connection).
# It allows the model to learn residual mappings, which helps in training deeper networks.
class ResConvBlock(nn.Module):
    def __init__(self, filters, kernel=3):
        """
        Initializes the Residual Convolutional Block.
        Args:
            filters: Number of filters (channels) for the convolutional layers.
            kernel: Kernel size for the convolutional layers.
        """
        super(ResConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(filters, filters, kernel)  # First convolution
        self.conv2 = nn.Conv2d(filters, filters, kernel)  # Second convolution
        self.pad = nn.ZeroPad2d(1)  # Adds padding to keep output size consistent
        self.norm1 = nn.BatchNorm2d(filters)  # Batch normalization for the first layer
        self.norm2 = nn.BatchNorm2d(filters)  # Batch normalization for the second layer
        self.lrelu = nn.LeakyReLU(0.1)  # Activation function with a small negative slope

    def forward(self, x):
        """
        Defines the forward pass of the block.
        Args:
            x: Input tensor.
        Returns:
            Tensor after passing through the residual block.
        """
        shortcut = x  # Save input as shortcut
        x = self.lrelu(self.norm1(self.pad(self.conv1(x))))  # First convolution with activation
        x = self.lrelu(self.norm2(self.pad(self.conv2(x))) + shortcut)  # Second convolution + shortcut
        return x


# Residual Bottleneck Block
# This block is designed to reduce the number of parameters while maintaining model performance.
class ResBottleNeck(nn.Module):
    def __init__(self, in_filters, out_filters, kernel=3):
        """
        Initializes the Residual Bottleneck Block.
        Args:
            in_filters: Number of input filters.
            out_filters: Number of output filters.
            kernel: Kernel size for the convolutional layers.
        """
        super(ResBottleNeck, self).__init__()
        self.conv1 = nn.Conv2d(in_filters, in_filters, kernel)  # Intermediate convolution
        self.conv2 = nn.Conv2d(in_filters, out_filters, 1)  # Output projection
        self.conv0 = nn.Conv2d(in_filters, in_filters, 1, stride=2)  # Input down-sampling
        self.conv = nn.Conv2d(in_filters, out_filters, 1, stride=2)  # Shortcut down-sampling
        self.pad = nn.ZeroPad2d(1)  # Adds padding for the convolutions
        self.norm0 = nn.BatchNorm2d(in_filters)  # Batch normalization for input
        self.norm1 = nn.BatchNorm2d(in_filters)  # Batch normalization for intermediate layer
        self.norm2 = nn.BatchNorm2d(out_filters)  # Batch normalization for output layer
        self.norm = nn.BatchNorm2d(out_filters)  # Batch normalization for shortcut
        self.lrelu = nn.LeakyReLU(0.1)  # Activation function

    def forward(self, x):
        """
        Defines the forward pass of the bottleneck block.
        Args:
            x: Input tensor.
        Returns:
            Tensor after passing through the bottleneck block.
        """
        shortcut = x  # Save input as shortcut
        x = self.lrelu(self.norm0(self.conv0(x)))  # Down-sample the input
        x = self.lrelu(self.norm1(self.pad(self.conv1(x))))  # Intermediate convolution
        x = self.lrelu(self.norm2(self.conv2(x)))  # Output projection
        shortcut = self.lrelu(self.norm(self.conv(shortcut)))  # Process shortcut
        return x + shortcut  # Add shortcut to the output


# Residual Block
# This block combines multiple ResConvBlocks and a ResBottleNeck.
class ResBlock(nn.Module):
    def __init__(self, in_filters, out_filters, kernel=3):
        """
        Initializes the Residual Block.
        Args:
            in_filters: Number of input filters.
            out_filters: Number of output filters.
        """
        super(ResBlock, self).__init__()
        self.conv0 = ResConvBlock(in_filters)  # First residual convolution block
        self.conv1 = ResConvBlock(in_filters)  # Second residual convolution block
        self.conv = ResBottleNeck(in_filters, out_filters)  # Residual bottleneck block

    def forward(self, x):
        """
        Defines the forward pass of the block.
        Args:
            x: Input tensor.
        Returns:
            Tensor after passing through the residual block.
        """
        return self.conv(self.conv1(self.conv0(x)))


# Start Block
# This block initializes the model and processes the input image.
class StartBlock(nn.Module):
    def __init__(self, filters):
        """
        Initializes the Start Block.
        Args:
            filters: Number of filters for the first convolutional layer.
        """
        super(StartBlock, self).__init__()
        self.conv1 = nn.Conv2d(3, filters, 7, stride=2)  # Initial convolution
        self.norm1 = nn.BatchNorm2d(filters)  # Batch normalization for the first layer
        self.lrelu = nn.LeakyReLU(0.1)  # Activation function

    def forward(self, x):
        """
        Defines the forward pass of the start block.
        Args:
            x: Input tensor (image).
        Returns:
            Tensor after initial processing.
        """
        return self.lrelu(self.norm1(self.conv1(x)))


# Global Max Pooling
# This module performs global average pooling on the feature maps.
class GMaxpool(nn.Module):
    def forward(self, x):
        """
        Performs global average pooling.
        Args:
            x: Input tensor.
        Returns:
            Tensor with pooled values.
        """
        return F.avg_pool2d(x, kernel_size=x.size()[2:])


# ResNet Architecture
# This class defines the entire ResNet model using the previously defined blocks.
class ResNet(nn.Module):
    def __init__(self, filters=16, dense_dim=256, num_classes=16):
        """
        Initializes the ResNet architecture.
        Args:
            filters: Number of filters for the initial layers.
            dense_dim: Number of neurons in the dense layer.
            num_classes: Number of output classes for classification.
        """
        super(ResNet, self).__init__()
        self.res0 = StartBlock(filters)  # Initial block
        self.res1 = ResBlock(filters, filters * 2)  # First residual block
        self.res2 = ResBlock(filters * 2, filters * 4)  # Second residual block
        self.res3 = ResBlock(filters * 4, filters * 8)  # Third residual block
        self.res4 = ResBlock(filters * 8, filters * 16)  # Fourth residual block
        self.res5 = ResBlock(filters * 16, filters * 32)  # Fifth residual block
        self.res6 = ResBlock(filters * 32, filters * 64)  # Sixth residual block

        self.avgpool = GMaxpool()  # Global average pooling
        self.flatten = nn.Flatten()  # Flatten the tensor
        self.dense1 = nn.Linear(1024, dense_dim)  # First dense layer
        self.dense2 = nn.Linear(dense_dim, num_classes)  # Second dense layer
        self.dropout = nn.Dropout(0.2)  # Dropout to prevent overfitting
        self.lrelu = nn.LeakyReLU(0.2)  # Activation function
        self.softmax = nn.Softmax(dim=1)  # Softmax for classification probabilities

    def forward(self, x):
        """
        Defines the forward pass of the ResNet model.
        Args:
            x: Input tensor (image).
        Returns:
            Classification probabilities for the input.
        """
        x = self.res0(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.res6(x)  # Feature extraction
        x = self.flatten(self.avgpool(x))  # Pooling and flattening
        x = self.softmax(self.dense2(self.lrelu(self.dropout(self.dense1(x)))))  # Dense layers with activation
        return x


# Instantiate and summarize the model
model = ResNet()
summary(model, (3, 750, 1000))  # Model summary for an input image of size (3, 750, 1000)
