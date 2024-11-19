import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

# model.py
class DoubleConv(nn.Module):
  """
  Double convolution block with Batch Normalization and ReLU activation.
  This block is used in both the downsampling and upsampling paths of the UNET.
  """
  def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False), # Convolution layer with 3x3 kernel, stride 1, padding 1
            nn.BatchNorm2d(out_channels), # Batch Normalization
            nn.ReLU(inplace=True), # ReLU activation
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False), # Second convolution layer
            nn.BatchNorm2d(out_channels), # Batch Normalization
            nn.ReLU(inplace=True) # ReLU activation
        )

  def forward(self, x):
    return self.conv(x)

class UNET(nn.Module):
    """
    UNET model architecture for image segmentation.
    """
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # Max pooling layer

        # Downsampling path
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Upsampling path
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature * 2, feature, kernel_size=2, stride=2 # Transpose convolution for upsampling
                )
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2) # Bottleneck layer
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1) # Final convolution layer

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1] # Reverse skip connections for concatenation

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            # If the shapes don't match, resize the tensor
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1) # Concatenate skip connection with upsampled tensor
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)

def test():
    x = torch.randn((3, 1, 161, 161)) # Create a random input tensor
    model = UNET(in_channels=1, out_channels=1) # Initialize the UNET model
    preds = model(x) # Get predictions from the model
    print(preds.shape) # Print the shape of the predictions
    print(x.shape) # Print the shape of the input tensor
    assert preds.shape == x.shape # Assert that the shapes of the predictions and input tensor are equal
