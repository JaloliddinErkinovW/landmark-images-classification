import torch
import torch.nn as nn
import torch.nn.init as init


# define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:

        super().__init__()

        # YOUR CODE HERE
        # Define a CNN architecture. Remember to use the variable num_classes
        # to size appropriately the output of your classifier, and if you use
        # the Dropout layer, use the variable "dropout" to indicate how much
        # to use (like nn.Dropout(p=dropout))

        self.conv1 = nn.Conv2d(3, 8, 3, padding = 1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding = 1)
        self.conv3 = nn.Conv2d(16, 32, 3, padding = 1)
        self.conv4 = nn.Conv2d(32, 64, 3, padding = 1)
        self.conv5 = nn.Conv2d(64, 128, 3, padding = 1)
        self.conv6 = nn.Conv2d(128, 256, 3, padding = 1)
        self.conv7 = nn.Conv2d(256, 1024, 3, padding = 1)
        
        self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(16)       
        self.bn3 = nn.BatchNorm2d(32)      
        self.bn4 = nn.BatchNorm2d(64)       
        self.bn5 = nn.BatchNorm2d(128)
        self.bn6 = nn.BatchNorm2d(256)
        self.bn7 = nn.BatchNorm2d(1024)
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.1)
        self.mpool = nn.MaxPool2d(2,2)
        self.apool = nn.AvgPool2d(1,1) 
        
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1024 * 7 * 7, 512, bias = True)
        self.dp = nn.Dropout(0.5)
        self.batchn1 = nn.BatchNorm1d(512)
        self.rl1 = nn.ReLU()
        
        self.fc2 = nn.Linear(512, num_classes)
        self.out = nn.LogSoftmax(dim = 1)
        
        self.apply(he_init)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # YOUR CODE HERE: process the input tensor through the
        # feature extractor, the pooling and the final linear
        # layers (if appropriate for the architecture chosen)
        
        
        x = self.apool(self.bn1(self.lrelu(self.conv1(x))))
        x = self.mpool(self.bn2(self.lrelu(self.conv2(x))))
        x = self.mpool(self.bn3(self.lrelu(self.conv3(x))))
        x = self.mpool(self.bn4(self.lrelu(self.conv4(x))))
        x = self.mpool(self.bn5(self.lrelu(self.conv5(x))))
        x = self.mpool(self.bn6(self.lrelu(self.conv6(x))))
        x = self.apool(self.bn7(self.lrelu(self.conv7(x))))
        
        
        x = self.flatten(x)
        
        x = self.dp(x)
        x = self.rl1(self.fc1(x))
        x = self.batchn1(x)
        x = self.dp(x)
        
        x = self.fc2(x)
        x = self.out(x)
        return x
    
def he_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            init.constant_(m.bias, 0)


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.next()

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
